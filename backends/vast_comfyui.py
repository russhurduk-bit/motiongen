"""
Vast.ai + ComfyUI backend for MotionGen.

Flow:
  1. Pick cheapest ok RTX 4090 offer (or reuse existing instance with label=motiongen)
  2. Create instance with vastai/comfy template + our provisioning.sh
  3. Wait until 18188/tcp is reachable
  4. Upload photo + video, patch workflow JSON, submit /prompt
  5. Stream progress via WS, download result to local
  6. DESTROY instance (unless keep_alive=True)

Requires workflows/default.json (or a path via arg).
Requires an optional public URL for provisioning.sh (served via local HTTP
when requested). If `provisioning_url=None`, uses vast.ai default provisioning.
"""
import asyncio, json, os, sys, time, requests
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backends import vast_helper as vh
from backends.comfy_client import ComfyClient
from workflows import workflow_parser as wp

BASE = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = BASE / "workflows"
DEFAULT_WORKFLOW = WORKFLOWS_DIR / "default.json"


def load_workflow(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_WORKFLOW
    if not path.exists():
        raise FileNotFoundError(
            f"No workflow at {path}. Положи workflow API JSON в {WORKFLOWS_DIR}/default.json")
    return json.loads(path.read_text())


async def find_or_create_instance(*, max_dph: float = 0.45,
                                  provisioning_url: str | None = None,
                                  prefer_reuse: bool = True) -> tuple[dict, bool]:
    """Return (instance, created_new_bool)."""
    # Reuse first live motiongen instance if present
    if prefer_reuse:
        for inst in vh.list_instances():
            if inst.get("label") == "motiongen" and inst.get("actual_status") == "running":
                return inst, False

    offers = vh.search_offers(max_dph=max_dph, limit=5)
    if not offers:
        raise RuntimeError(
            f"No RTX 4090 offers under ${max_dph}/hr. Попробуй повысить лимит.")
    offer = offers[0]
    resp = vh.create_instance(offer["id"], provisioning_url=provisioning_url)
    iid = resp.get("new_contract") or resp.get("id")
    if not iid:
        raise RuntimeError(f"Bad create response: {resp}")
    inst = vh.wait_running(iid, timeout=1200)
    return inst, True


async def wait_comfy_ready(endpoint: str, *, timeout: int = 1800, poll: int = 15,
                           on_progress=None):
    """Wait until ComfyUI /system_stats responds (provisioning может занять 10-30 мин)."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{endpoint}/system_stats", timeout=8)
            if r.ok:
                return r.json()
        except Exception:
            pass
        if on_progress:
            await on_progress(
                f"ComfyUI ещё не поднялся ({int(time.time()-start)}s)…")
        await asyncio.sleep(poll)
    raise TimeoutError(f"ComfyUI not ready at {endpoint} in {timeout}s")


async def generate(*,
                   photo_path: str,
                   video_path: str | None = None,
                   prompt: str = "",
                   negative_prompt: str = "",
                   workflow_path: str | Path | None = None,
                   out_dir: str = str(BASE / "output"),
                   on_progress=None,
                   keep_alive: bool = False,
                   provisioning_url: str | None = None,
                   max_dph: float = 0.45) -> dict:
    """
    Returns dict {
        final_path, cost_usd_estimate, instance_id, total_seconds,
        comfy_endpoint, workflow_classes, missing_classes (if any)
    }
    """
    async def p(msg):
        if on_progress:
            try: await on_progress(msg)
            except Exception: pass

    wf_template = load_workflow(workflow_path)
    classes = wp.classes_used(wf_template)

    await p("Ищу и арендую RTX 4090 на vast.ai…")
    inst, created = await find_or_create_instance(
        max_dph=max_dph, provisioning_url=provisioning_url)
    iid = inst["id"]
    dph = inst.get("dph_total") or 0
    t_start = time.time()

    try:
        endpoint = vh.public_endpoint(inst, 18188)
        await p(f"Инстанс {iid} готов ({dph:.2f} $/hr). Жду ComfyUI по {endpoint}…")
        await wait_comfy_ready(endpoint, on_progress=on_progress)

        c = ComfyClient(endpoint, job_timeout=1800)
        object_info = c.object_info()
        missing = wp.validate_against_object_info(wf_template, object_info)
        if missing:
            raise RuntimeError(
                f"Workflow requires custom nodes not installed on this instance: {missing}. "
                f"Добавь их в provisioning.sh.")

        await p("Загружаю фото в ComfyUI…")
        img_name = c.upload_image(photo_path, kind="input")
        vid_name = None
        if video_path:
            await p("Загружаю видео в ComfyUI…")
            vid_name = c.upload_image(video_path, kind="input")

        patched = wp.patch(wf_template,
                           image_name=img_name, video_name=vid_name,
                           prompt=prompt, negative=negative_prompt)

        await p("Отправляю workflow на исполнение…")
        pid = c.submit(patched)
        hist = await c.wait(pid, on_progress=on_progress)

        await p("Скачиваю результат…")
        saved = c.collect_outputs(hist, out_dir)
        if not saved:
            raise RuntimeError("No outputs in history")

        elapsed = time.time() - t_start
        cost_estimate = dph * elapsed / 3600
        return {
            "final_path": saved[-1],
            "all_outputs": saved,
            "cost_usd_estimate": round(cost_estimate, 4),
            "instance_id": iid,
            "total_seconds": int(elapsed),
            "comfy_endpoint": endpoint,
            "workflow_classes": sorted(classes),
            "created_instance": created,
        }
    finally:
        if not keep_alive and created:
            try:
                await p("Гашу инстанс (keep_alive=False)…")
                vh.destroy_instance(iid)
            except Exception as e:
                await p(f"⚠️ destroy failed: {e}")


# CLI smoke test (без реальной генерации — только validate workflow)
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    ap.add_argument("--photo")
    ap.add_argument("--video")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--negative", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="только parse + find offer, без create")
    args = ap.parse_args()

    wf = load_workflow(args.workflow)
    print(f"Workflow classes: {sorted(wp.classes_used(wf))}")
    print(f"Image node: {wp.find_image_node(wf)}")
    print(f"Video node: {wp.find_video_node(wf)}")
    print(f"Prompt nodes (pos, neg): {wp.find_prompt_nodes(wf)}")
    print(f"Samplers: {wp.list_sampler_nodes(wf)}")

    if args.dry_run:
        offers = vh.search_offers(max_dph=0.45, limit=3)
        print(f"\nTop 3 offers under $0.45/hr:")
        for o in offers:
            print(f"  id={o['id']}  {o['gpu_name']}  ${o['dph_total']:.3f}/hr  "
                  f"disk={o['disk_space']}GB  rel={o.get('expected_reliability',0):.2f}")
        print(f"\nUser balance: {vh.user().get('credit', '?')}")
        sys.exit(0)

    if not args.photo:
        print("--photo required (or use --dry-run)"); sys.exit(1)

    async def go():
        async def pr(m): print("  »", m)
        out = await generate(photo_path=args.photo, video_path=args.video,
                             prompt=args.prompt, negative_prompt=args.negative,
                             workflow_path=args.workflow,
                             on_progress=pr, keep_alive=False)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    asyncio.run(go())
