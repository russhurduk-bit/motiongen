"""
Higgsfield browser backend.
Connects to existing persistent profile, opens /create/video, switches to
Motion Control tab, uploads image, picks preset, types prompt, clicks Generate,
waits for result in History, downloads MP4.

Designed to be called once per generation. Profile stays open between calls
if a long-lived browser is kept alive (use keep_alive() helper).
"""
import asyncio, os, time, json, re
from pathlib import Path
os.environ.setdefault("DISPLAY", ":98")
from playwright.async_api import async_playwright, BrowserContext, Page

PROFILE = os.environ.get(
    "HF_PROFILE_DIR", str(Path(__file__).resolve().parent.parent / "browser_profile"))
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
CREATE_URL = "https://higgsfield.ai/create/video"


async def open_context(pw):
    return await pw.chromium.launch_persistent_context(
        user_data_dir=PROFILE, headless=False,
        args=["--display=:98", "--no-sandbox", "--disable-dev-shm-usage",
              "--disable-blink-features=AutomationControlled",
              "--window-size=1920,1080"],
        viewport={"width": 1920, "height": 1080},
        user_agent=UA, ignore_https_errors=True,
    )


async def is_logged_in(page: Page) -> bool:
    try:
        body = (await page.inner_text("body")).lower()
    except Exception:
        return False
    return ("log in" not in body and "sign up" not in body
            and ("create" in body or "history" in body))


async def goto_motion_control(page: Page):
    await page.goto(CREATE_URL, wait_until="domcontentloaded", timeout=60000)
    await asyncio.sleep(4)
    # Click Motion Control tab in sidebar
    for sel in ['button:has-text("Motion Control")',
                'div[role="tab"]:has-text("Motion Control")',
                'a:has-text("Motion Control")']:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=2000):
                await el.click()
                await asyncio.sleep(2)
                return True
        except Exception:
            continue
    return False


async def upload_start_frame(page: Page, image_path: str):
    """Upload to Start frame slot."""
    # The Start frame card has hidden file input nearby
    # Try clicking the slot first
    try:
        await page.locator('text=Start frame').first.click(timeout=3000)
        await asyncio.sleep(1)
    except Exception:
        pass
    # Fill the next visible file input
    file_inputs = await page.locator('input[type="file"]').all()
    for fi in file_inputs:
        try:
            await fi.set_input_files(image_path)
            await asyncio.sleep(2)
            return True
        except Exception:
            continue
    return False


async def pick_preset(page: Page, preset_name: str):
    """Open preset gallery and click the named preset."""
    # Try common entry points
    for sel in [f'text="{preset_name}"',
                f'div:has-text("{preset_name}"):not(:has(div))']:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=2500):
                await el.click()
                await asyncio.sleep(1.5)
                return True
        except Exception:
            continue
    return False


async def type_prompt(page: Page, prompt: str):
    ta = page.locator('textarea').first
    await ta.wait_for(state="visible", timeout=10000)
    await ta.fill("")
    await ta.type(prompt, delay=15)


async def click_generate(page: Page):
    btn = page.locator('button:has-text("Generate")').first
    await btn.wait_for(state="visible", timeout=10000)
    await btn.click()


async def wait_and_download(page: Page, out_dir: str, timeout=600) -> str | None:
    """Open History panel and wait for the latest MP4 download link."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        await page.locator('button:has-text("History")').first.click(timeout=5000)
    except Exception:
        pass
    deadline = time.time() + timeout
    saved = None
    async with page.expect_download(timeout=timeout * 1000) as dl_info:
        # Poll for a "Download" button on the newest job card
        while time.time() < deadline and saved is None:
            try:
                await page.locator('button:has-text("Download")').first.click(timeout=4000)
                break
            except Exception:
                await asyncio.sleep(5)
        dl = await dl_info.value
    fname = dl.suggested_filename or f"hf_{int(time.time())}.mp4"
    dst = os.path.join(out_dir, fname)
    await dl.save_as(dst)
    return dst


# Smoke test entrypoint
async def smoke():
    async with async_playwright() as pw:
        ctx = await open_context(pw)
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
        await page.goto(CREATE_URL, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(4)
        ok = await is_logged_in(page)
        print("logged_in:", ok)
        if ok:
            await goto_motion_control(page)
            await page.screenshot(path="/tmp/hf_motion_tab.png", full_page=True)
            print("screenshot: /tmp/hf_motion_tab.png")
        await ctx.close()


if __name__ == "__main__":
    asyncio.run(smoke())
