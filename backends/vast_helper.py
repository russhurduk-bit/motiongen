"""
Vast.ai wrapper — search, create, wait-for-running, destroy.
Uses the single REST API at console.vast.ai. Authenticated via Bearer API_KEY.
Key resolution: env VAST_API_KEY, then env VAST_API_KEY_FILE, then
~/.config/vastai/vast_api_key.
"""
import json, os, time, requests

API = "https://console.vast.ai/api/v0"


def _key() -> str:
    k = os.environ.get("VAST_API_KEY")
    if k:
        return k.strip()
    for p in (os.environ.get("VAST_API_KEY_FILE"),
              os.path.expanduser("~/.config/vastai/vast_api_key")):
        if p and os.path.exists(p):
            return open(p).read().strip()
    raise RuntimeError("No VAST_API_KEY / key file found")


def _h() -> dict:
    return {"Authorization": f"Bearer {_key()}",
            "Accept": "application/json",
            "Content-Type": "application/json"}


# ---------------- search / offers ------------------

def search_offers(*, gpu="RTX 4090", num_gpus=1, max_dph=0.5,
                  min_reliability=0.97, min_disk_gb=80,
                  min_inet_down=300, limit=10,
                  extra_filters: dict | None = None) -> list[dict]:
    """Search offers via console.vast.ai bundles endpoint.
    Vast.ai expects JSON body with operator dicts, not URL query string.
    """
    body = {
        "verified": {"eq": True},
        "external": {"eq": False},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "gpu_name": {"eq": gpu},
        "num_gpus": {"eq": str(num_gpus)},
        "dph_total": {"lt": str(max_dph)},
        "reliability": {"gte": str(min_reliability)},
        "disk_space": {"gte": str(min_disk_gb)},
        "inet_down": {"gte": str(min_inet_down)},
        "order": [["dph_total", "asc"]],
        "type": "on-demand",
        "allocated_storage": 5.0,
    }
    if extra_filters:
        body.update(extra_filters)
    r = requests.post(f"{API}/bundles/", headers=_h(),
                      data=json.dumps(body), timeout=30)
    if not r.ok:
        # fallback: try GET with q param (older CLI style)
        r = requests.get(f"{API}/bundles/", headers=_h(),
                         params={"q": json.dumps(body)}, timeout=30)
    r.raise_for_status()
    return (r.json().get("offers") or [])[:limit]


# ---------------- lifecycle ------------------------

def create_instance(offer_id: int, *,
                    image: str = "vastai/comfy",
                    disk_gb: int = 80,
                    label: str = "motiongen",
                    provisioning_url: str | None = None,
                    extra_env: dict | None = None,
                    volume_id: int | None = None,
                    onstart: str | None = None) -> dict:
    """Create new instance from an offer. Returns API response (includes new_contract)."""
    env = {
        "-p 18188:18188": None, "-p 8288:8288": None,
        "-p 8188:8188":   None, "-p 8080:8080": None, "-p 1111:1111": None,
        "-e COMFYUI_ARGS": "--disable-auto-launch --port 18188 --enable-cors-header",
        "-e COMFYUI_API_BASE": "http://localhost:18188",
        "-e DATA_DIRECTORY": "/workspace/",
        "-e OPEN_BUTTON_PORT": "1111",
        "-e OPEN_BUTTON_TOKEN": "1",
    }
    if provisioning_url:
        env["-e PROVISIONING_SCRIPT"] = provisioning_url
    if extra_env:
        env.update(extra_env)

    # Vast.ai expects env as flat string with -p/-e tokens; flatten our dict
    env_str = " ".join(
        k if v is None else f'{k}="{v}"' for k, v in env.items()
    )

    body = {
        "client_id": "motiongen",
        "image": image,
        "disk": disk_gb,
        "runtype": "ssh_proxy",
        "label": label,
        "env": env_str,
    }
    if onstart:
        body["onstart"] = onstart
    if volume_id:
        body["volume_id"] = volume_id

    r = requests.put(f"{API}/asks/{offer_id}/", headers=_h(),
                     data=json.dumps(body), timeout=30)
    r.raise_for_status()
    return r.json()


def list_instances() -> list[dict]:
    r = requests.get(f"{API}/instances/", headers=_h(), timeout=30)
    r.raise_for_status()
    return r.json().get("instances") or []


def show_instance(iid: int) -> dict | None:
    for inst in list_instances():
        if inst["id"] == iid:
            return inst
    return None


def destroy_instance(iid: int) -> dict:
    r = requests.delete(f"{API}/instances/{iid}/", headers=_h(), timeout=30)
    r.raise_for_status()
    return r.json()


def wait_running(iid: int, *, timeout: int = 900, poll: int = 10) -> dict:
    """Wait until instance is running AND ports are mapped AND 18188 is open."""
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        inst = show_instance(iid)
        if inst:
            st = inst.get("actual_status")
            msg = inst.get("status_msg") or ""
            if st != last_status:
                last_status = st
            if "error" in msg.lower():
                raise RuntimeError(f"instance {iid} error: {msg}")
            if st == "running" and (inst.get("ports") or {}).get("18188/tcp"):
                return inst
        time.sleep(poll)
    raise TimeoutError(f"instance {iid} not running within {timeout}s")


def public_endpoint(inst: dict, internal_port: int = 18188) -> str:
    ports = inst.get("ports") or {}
    mapping = ports.get(f"{internal_port}/tcp")
    if not mapping:
        raise RuntimeError(f"no public mapping for {internal_port}")
    host = inst.get("public_ipaddr") or inst.get("ssh_host")
    return f"http://{host}:{mapping[0]['HostPort']}"


# ---------------- billing ------------------------

def user() -> dict:
    r = requests.get(f"{API}/users/current/", headers=_h(), timeout=15)
    r.raise_for_status()
    return r.json()
