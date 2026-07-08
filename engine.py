"""
MotionGen engine — single entry point: generate_motion(photo_path, video_path)

Built-in post-process rules:
1. Audio from ref_video is muxed into the final result.
2. Result aspect ratio is aligned to the photo reference (scale+pad if needed).
3. Final path returned is ready to be sent as a Telegram DOCUMENT (not video).
"""
import asyncio, os, sys, time, json, re, subprocess, shlex
from pathlib import Path
os.environ.setdefault("DISPLAY", ":98")
from playwright.async_api import async_playwright

BASE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = str(BASE / "output")
# Persistent browser profile for the Higgsfield backend (override via env).
PROFILE = os.environ.get("HF_PROFILE_DIR", str(BASE / "browser_profile"))
URL = "https://higgsfield.ai/create/motion-control"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


# ----------------------------- helpers ---------------------------------

def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def ffprobe_size(path: str) -> tuple[int, int]:
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", path])
    w, h = p.stdout.strip().split(",")[:2]
    return int(w), int(h)


def ffprobe_duration(path: str) -> float:
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path])
    return float(p.stdout.strip())


def transcode_h264_keep_audio(src: str, dst: str) -> str:
    """Higgsfield chokes on HEVC. Re-encode to H.264 baseline, KEEP audio."""
    run(["ffmpeg", "-y", "-i", src,
         "-c:v", "libx264", "-profile:v", "baseline", "-level", "4.0",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-c:a", "aac", "-b:a", "192k",  # keep audio, re-encode to aac
         dst])
    return dst


def estimate_cost(video_duration_s: float) -> int:
    """Higgsfield Motion Control cost formula (empirical):
       base 7 credits (for up to 5 sec) + 1 credit per extra second.
       For 6.97s → 7 + 2 = 9? observed 13.
       Safer model observed: 7 + (duration - 5) * ceil factor ≈ ceil(duration - 5) rounded-up.
       In practice 6.97s gave 13, so formula: 7 + (ceil(duration) - 5) * k
       With k=3: 7 + 2*3 = 13 ✓
    """
    import math
    extra = max(0, math.ceil(video_duration_s) - 5)
    return 7 + extra * 3


def mux_audio(result_silent: str, source_audio: str, out: str) -> str:
    """Take video from result_silent, audio from source_audio."""
    run(["ffmpeg", "-y", "-i", result_silent, "-i", source_audio,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-map", "0:v:0", "-map", "1:a:0?", "-shortest",
         "-movflags", "+faststart", out])
    return out


def align_aspect_to_photo(video: str, photo: str, out: str) -> str:
    """If video aspect != photo aspect, scale+pad the video to photo's aspect (no crop)."""
    pw, ph = ffprobe_size(photo)
    vw, vh = ffprobe_size(video)
    photo_ar = pw / ph
    video_ar = vw / vh
    if abs(photo_ar - video_ar) < 0.01:
        # Already aligned; just copy (rename)
        if os.path.abspath(video) != os.path.abspath(out):
            run(["cp", "-f", video, out])
        return out
    # Pad video to photo_ar. Keep video height; compute new width, pad sides/top.
    if photo_ar > video_ar:
        # photo is wider; pad horizontally
        new_w = int(round(vh * photo_ar / 2) * 2)
        new_h = int(round(vh / 2) * 2)
    else:
        # photo is taller; pad vertically
        new_w = int(round(vw / 2) * 2)
        new_h = int(round(vw / photo_ar / 2) * 2)
    vf = (f"scale={vw}:{vh}:force_original_aspect_ratio=decrease,"
          f"pad={new_w}:{new_h}:({new_w}-iw)/2:({new_h}-ih)/2:black")
    run(["ffmpeg", "-y", "-i", video, "-vf", vf, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "copy",
         "-movflags", "+faststart", out])
    return out


# ----------------------------- browser flow ----------------------------

async def _ensure_browser(pw):
    return await pw.chromium.launch_persistent_context(
        user_data_dir=PROFILE, headless=False, accept_downloads=True,
        args=["--display=:98", "--no-sandbox", "--disable-dev-shm-usage",
              "--disable-blink-features=AutomationControlled",
              "--window-size=1920,1080"],
        viewport={"width": 1920, "height": 1080},
        user_agent=UA, ignore_https_errors=True,
    )


async def _click_if_visible(page, selectors, timeout=2500):
    for sel in selectors:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=timeout):
                await el.click()
                return sel
        except Exception:
            continue
    return None


async def _get_cost(page) -> int | None:
    body = await page.inner_text("body")
    m = re.search(r"Generate\s+(\d+)", body)
    return int(m.group(1)) if m else None


async def _run_higgsfield(video_path: str, photo_path: str, *,
                          work_dir: str, on_progress=None):
    """Return downloaded silent MP4 path + metadata dict."""
    async def progress(msg):
        if on_progress:
            try: await on_progress(msg)
            except Exception: pass

    async with async_playwright() as pw:
        ctx = await _ensure_browser(pw)
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")

        # XHR capture for the result URL
        result_urls: list[str] = []

        async def on_resp(resp):
            try:
                ct = (resp.headers.get("content-type") or "").lower()
                if "json" not in ct: return
                u = resp.url
                if "higgsfield" not in u: return
                if resp.status >= 400: return
                txt = await resp.text()
                if "d8j0ntlcm91z4.cloudfront.net" in txt:
                    for m in re.finditer(
                        r'https://d8j0ntlcm91z4\.cloudfront\.net/[^\s"\'<>]+\.mp4', txt):
                        if m.group(0) not in result_urls:
                            result_urls.append(m.group(0))
            except Exception:
                pass
        page.on("response", lambda r: asyncio.create_task(on_resp(r)))

        await progress("Открываю Higgsfield Motion Control…")
        await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(6)

        # Session health check — detect expired login, abort early with clear error
        body_lc = (await page.inner_text("body")).lower()
        if "log in" in body_lc and "sign up" in body_lc:
            raise RuntimeError(
                "Higgsfield session expired — re-run the login helper and retry. "
                "(Automatic login is disabled because it needs an email code.)")

        # Upload video + photo
        inputs = await page.locator('input[type="file"]').all()
        if len(inputs) < 2:
            raise RuntimeError(f"Expected 2 file inputs, got {len(inputs)}")
        await progress("Загружаю видео-референс…")
        await inputs[0].set_input_files(video_path)
        await asyncio.sleep(4)
        inputs = await page.locator('input[type="file"]').all()
        await progress("Загружаю фото-референс…")
        await (inputs[1] if len(inputs) >= 2 else inputs[0]).set_input_files(photo_path)
        await asyncio.sleep(4)

        # Accept agreement
        await _click_if_visible(page, ['button:has-text("I agree, continue")'])
        await asyncio.sleep(4)

        # Check warnings
        body = (await page.inner_text("body")).lower()
        if "too low" in body or "0 pixels" in body:
            raise RuntimeError("Upload warning present — нужно перекодировать видео в H264.")

        cost = await _get_cost(page)
        if cost is None or cost > 80:
            raise RuntimeError(f"Unexpected Generate cost: {cost}")

        await progress(f"Всё готово, запускаю генерацию (~{cost} кредитов)…")
        await page.locator('button:has-text("Generate")').first.click()
        t0 = time.time()
        await asyncio.sleep(6)

        # Open History
        await _click_if_visible(page, ['button:has-text("History")'])
        await asyncio.sleep(3)

        # Poll up to 20 min
        deadline = t0 + 1200
        last_tick = 0
        while time.time() < deadline:
            await asyncio.sleep(10)
            elapsed = int(time.time() - t0)

            # Trigger an interaction to refresh the history list (click area)
            try:
                await page.locator('aside img:visible').first.click(timeout=1500)
            except Exception:
                pass

            if result_urls:
                break

            if elapsed - last_tick >= 60:
                await progress(f"Рендер идёт… {elapsed//60} мин прошло, жду результат")
                last_tick = elapsed

        if not result_urls:
            raise TimeoutError("Higgsfield took >20 min — aborting")

        # Pick latest (should be our generation — today's timestamp)
        today_tag = time.strftime("%Y%m%d")
        fresh = [u for u in result_urls if today_tag in u]
        target = (fresh or result_urls)[-1]
        await progress("Результат получен, скачиваю…")

        raw_path = os.path.join(work_dir, f"raw_{int(time.time())}.mp4")
        api = ctx.request
        resp = await api.get(target, headers={"User-Agent": UA,
                                              "Referer": "https://higgsfield.ai/"})
        if not resp.ok:
            raise RuntimeError(f"Download failed: HTTP {resp.status}")
        body_bytes = await resp.body()
        with open(raw_path, "wb") as f:
            f.write(body_bytes)

        meta = {
            "cost_credits": cost,
            "render_seconds": int(time.time() - t0),
            "url": target,
            "raw_path": raw_path,
            "raw_size": len(body_bytes),
        }
        await ctx.close()
        return meta


# ----------------------------- public API ------------------------------

async def generate_motion(photo_path: str, video_path: str, *,
                          work_dir: str = DEFAULT_OUTPUT,
                          on_progress=None):
    """
    Full pipeline. Returns dict with keys:
      final_path, cost_credits, render_seconds, total_seconds,
      result_w, result_h, photo_w, photo_h, ref_video_duration.
    """
    os.makedirs(work_dir, exist_ok=True)
    t_total = time.time()

    # Pre-flight checks
    ref_dur = ffprobe_duration(video_path)
    if ref_dur < 3 or ref_dur > 30:
        raise ValueError(
            f"Video duration {ref_dur:.1f}s out of [3, 30] sec range required by Kling")
    ph_w, ph_h = ffprobe_size(photo_path)

    # Transcode video (HEVC→H264) if needed, preserving audio
    async def progress(m):
        if on_progress:
            try: await on_progress(m)
            except Exception: pass

    await progress("Перекодирую видео в H264 (со звуком)…")
    h264_path = os.path.join(work_dir, f"ref_h264_{int(time.time())}.mp4")
    transcode_h264_keep_audio(video_path, h264_path)

    # Higgsfield generation (silent result)
    meta = await _run_higgsfield(h264_path, photo_path,
                                 work_dir=work_dir, on_progress=on_progress)

    # Post-process: aspect alignment
    await progress("Пост-обработка: выравниваю формат по фото…")
    aligned = os.path.join(work_dir, f"aligned_{int(time.time())}.mp4")
    align_aspect_to_photo(meta["raw_path"], photo_path, aligned)

    # Post-process: mux audio from the ORIGINAL ref video (keeps the original soundtrack)
    await progress("Добавляю звук из видео-референса…")
    final = os.path.join(work_dir, f"motiongen_{int(time.time())}.mp4")
    mux_audio(aligned, video_path, final)

    rw, rh = ffprobe_size(final)
    return {
        "final_path": final,
        "cost_credits": meta["cost_credits"],
        "render_seconds": meta["render_seconds"],
        "total_seconds": int(time.time() - t_total),
        "result_w": rw, "result_h": rh,
        "photo_w": ph_w, "photo_h": ph_h,
        "ref_video_duration": ref_dur,
    }


async def generate_via_vast(photo_path: str, video_path: str | None = None, *,
                            prompt: str = "", negative_prompt: str = "",
                            work_dir: str = DEFAULT_OUTPUT,
                            on_progress=None, keep_alive: bool = False,
                            workflow_path: str | None = None,
                            provisioning_url: str | None = None):
    """Vast.ai + ComfyUI backend. Expects a workflow JSON at workflows/default.json."""
    from backends import vast_comfyui as vc
    out = await vc.generate(
        photo_path=photo_path, video_path=video_path,
        prompt=prompt, negative_prompt=negative_prompt,
        workflow_path=workflow_path, out_dir=work_dir,
        on_progress=on_progress, keep_alive=keep_alive,
        provisioning_url=provisioning_url)

    async def _progress(m):
        if on_progress:
            try: await on_progress(m)
            except Exception: pass

    # Post-process: audio mux + aspect align (if video reference provided)
    if video_path:
        await _progress("Пост-обработка: aspect + audio…")
        aligned = os.path.join(work_dir, f"aligned_vast_{int(time.time())}.mp4")
        align_aspect_to_photo(out["final_path"], photo_path, aligned)
        final = os.path.join(work_dir, f"motiongen_vast_{int(time.time())}.mp4")
        mux_audio(aligned, video_path, final)
        out["final_path"] = final

    return out


# CLI smoke test
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--video", required=True)
    args = ap.parse_args()

    async def progress(m): print(f"[progress] {m}")
    out = asyncio.run(generate_motion(args.photo, args.video, on_progress=progress))
    print(json.dumps(out, indent=2, ensure_ascii=False))
