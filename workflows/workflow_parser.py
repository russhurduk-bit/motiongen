"""
Auto-patch ComfyUI workflow API JSON.
Detects roles of nodes by class_type and _meta.title, replaces inputs with given
image/video/prompts/seed. Returns a new (deep-copied) workflow ready for /prompt.
"""
import json, random, re

IMG_LOADERS = {"LoadImage", "LoadImageMask", "LoadImagePath",
               "ETN_LoadImageBase64", "Image Load"}
VID_LOADERS = {"VHS_LoadVideo", "VHS_LoadVideoPath", "LoadVideo",
               "LoadVideoPath", "WanVideoLoadVideo"}
TEXT_ENCODERS = {"CLIPTextEncode", "CLIPTextEncodeSDXL",
                 "WanVideoTextEncode", "T5TextEncode", "smZ CLIPTextEncode"}
SAMPLERS = {"KSampler", "KSamplerAdvanced", "SamplerCustom",
            "SamplerCustomAdvanced", "WanVideoSampler"}


def _nodes(wf: dict):
    """Iterate only over valid node entries (skip comment keys)."""
    for nid, n in wf.items():
        if isinstance(n, dict) and "class_type" in n:
            yield nid, n


def clone(wf: dict) -> dict:
    cleaned = {nid: n for nid, n in _nodes(wf)}
    return json.loads(json.dumps(cleaned))


def find_prompt_nodes(wf: dict) -> tuple[str | None, str | None]:
    """Return (positive_node_id, negative_node_id).
    Strategy: look at KSampler.positive/negative links first (most reliable).
    Fallback: _meta.title heuristic."""
    pos = neg = None
    for nid, n in _nodes(wf):
        if n.get("class_type") in SAMPLERS:
            p = (n.get("inputs") or {}).get("positive")
            q = (n.get("inputs") or {}).get("negative")
            if isinstance(p, list) and not pos:
                pos = str(p[0])
            if isinstance(q, list) and not neg:
                neg = str(q[0])
    if pos and neg:
        return pos, neg

    for nid, n in _nodes(wf):
        if n.get("class_type") not in TEXT_ENCODERS:
            continue
        title = (n.get("_meta", {}).get("title", "") or "").lower()
        if "neg" in title and not neg:
            neg = nid
        elif ("pos" in title or title == "") and not pos:
            pos = nid
    return pos, neg


def find_image_node(wf: dict, role_hint: str | None = None) -> str | None:
    candidates = [(nid, n) for nid, n in _nodes(wf)
                  if n.get("class_type") in IMG_LOADERS]
    if not candidates:
        return None
    if role_hint:
        for nid, n in candidates:
            title = (n.get("_meta", {}).get("title", "") or "").lower()
            if role_hint.lower() in title:
                return nid
    return candidates[0][0]


def find_video_node(wf: dict) -> str | None:
    for nid, n in _nodes(wf):
        if n.get("class_type") in VID_LOADERS:
            return nid
    return None


def list_sampler_nodes(wf: dict) -> list[str]:
    return [nid for nid, n in _nodes(wf) if n.get("class_type") in SAMPLERS]


def classes_used(wf: dict) -> set[str]:
    return {n.get("class_type") for _, n in _nodes(wf)}


def patch(workflow: dict, *,
          image_name: str | None = None,
          video_name: str | None = None,
          prompt: str | None = None,
          negative: str | None = None,
          seed: int | None = None,
          role_hint: str = "reference") -> dict:
    wf = clone(workflow)

    if image_name:
        nid = find_image_node(wf, role_hint)
        if nid is not None:
            wf[nid]["inputs"]["image"] = image_name

    if video_name:
        nid = find_video_node(wf)
        if nid is not None:
            inp = wf[nid]["inputs"]
            if "video" in inp:
                inp["video"] = video_name
            elif "video_path" in inp:
                inp["video_path"] = video_name
            else:
                inp["video"] = video_name

    pos_id, neg_id = find_prompt_nodes(wf)
    if prompt is not None and pos_id is not None:
        wf[pos_id]["inputs"]["text"] = prompt
    if negative is not None and neg_id is not None:
        wf[neg_id]["inputs"]["text"] = negative

    final_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    for nid in list_sampler_nodes(wf):
        if "seed" in wf[nid]["inputs"]:
            wf[nid]["inputs"]["seed"] = final_seed

    return wf


def validate_against_object_info(wf: dict, object_info: dict) -> list[str]:
    """Return list of missing class_types on the target instance."""
    missing = []
    for nid, n in _nodes(wf):
        ct = n.get("class_type")
        if ct and ct not in object_info:
            missing.append(ct)
    return missing
