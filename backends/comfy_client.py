"""
ComfyUI REST client — talks to /prompt, /history, /upload/image, /view, /ws, /object_info.
"""
import asyncio, json, os, time, uuid, requests, websockets
from urllib.parse import urlencode


class ComfyClient:
    def __init__(self, host: str, *, http_timeout: int = 60, job_timeout: int = 1800):
        self.host = host.rstrip("/")        # e.g. http://1.2.3.4:23456
        self.ws_host = self.host.replace("http://", "ws://").replace("https://", "wss://")
        self.http_timeout = http_timeout
        self.job_timeout = job_timeout
        self.client_id = str(uuid.uuid4())

    # ---------------- sync-ish REST ----------------
    def stats(self) -> dict:
        r = requests.get(f"{self.host}/system_stats", timeout=15)
        r.raise_for_status()
        return r.json()

    def object_info(self) -> dict:
        r = requests.get(f"{self.host}/object_info", timeout=30)
        r.raise_for_status()
        return r.json()

    def upload_image(self, path: str, *, subfolder: str = "", kind: str = "input") -> str:
        with open(path, "rb") as f:
            files = {"image": (os.path.basename(path), f, "application/octet-stream")}
            data = {"subfolder": subfolder, "type": kind, "overwrite": "true"}
            r = requests.post(f"{self.host}/upload/image", files=files, data=data,
                              timeout=self.http_timeout)
        r.raise_for_status()
        return r.json()["name"]

    def submit(self, workflow_api: dict, *, extra_data: dict | None = None) -> str:
        body = {"prompt": workflow_api, "client_id": self.client_id}
        if extra_data:
            body["extra_data"] = extra_data
        r = requests.post(f"{self.host}/prompt", json=body, timeout=30)
        r.raise_for_status()
        return r.json()["prompt_id"]

    def history(self, prompt_id: str) -> dict | None:
        r = requests.get(f"{self.host}/history/{prompt_id}", timeout=30)
        if not r.ok:
            return None
        data = r.json()
        return data.get(prompt_id)

    def interrupt(self):
        requests.post(f"{self.host}/interrupt", timeout=10)

    def free_memory(self):
        requests.post(f"{self.host}/free", json={"unload_models": True, "free_memory": True},
                      timeout=15)

    def view(self, filename: str, subfolder: str = "", type_: str = "output") -> bytes:
        qs = urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
        r = requests.get(f"{self.host}/view?{qs}", timeout=300, stream=True)
        r.raise_for_status()
        return r.content

    # ---------------- async progress via WS ---------
    async def wait(self, prompt_id: str, *, on_progress=None) -> dict:
        """Wait for the submitted prompt to complete via WS; return history outputs."""
        ws_url = f"{self.ws_host}/ws?clientId={self.client_id}"
        deadline = time.time() + self.job_timeout
        async with websockets.connect(ws_url, max_size=None) as ws:
            while True:
                if time.time() > deadline:
                    raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out")
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                except asyncio.TimeoutError:
                    continue
                if isinstance(raw, (bytes, bytearray)):
                    continue  # binary preview — ignore
                msg = json.loads(raw)
                t = msg.get("type")
                d = msg.get("data", {})
                if t == "progress" and on_progress:
                    try:
                        await on_progress(f"ComfyUI: {d.get('value')}/{d.get('max')}")
                    except Exception:
                        pass
                elif t == "execution_error" and d.get("prompt_id") == prompt_id:
                    raise RuntimeError(f"ComfyUI error: {d}")
                elif t == "executing" and d.get("prompt_id") == prompt_id \
                        and d.get("node") is None:
                    break
        hist = self.history(prompt_id)
        if not hist:
            raise RuntimeError(f"No history for {prompt_id}")
        return hist

    # ---------------- download all outputs ----------
    def collect_outputs(self, hist: dict, out_dir: str) -> list[str]:
        os.makedirs(out_dir, exist_ok=True)
        saved = []
        for node_id, out in (hist.get("outputs") or {}).items():
            for kind in ("videos", "gifs", "images"):
                for item in out.get(kind) or []:
                    data = self.view(item["filename"],
                                     item.get("subfolder", ""),
                                     item.get("type", "output"))
                    dst = os.path.join(out_dir, item["filename"])
                    with open(dst, "wb") as f:
                        f.write(data)
                    saved.append(dst)
        return saved
