"""
Build custom_v2_car_dance workflow based on the real ofmhub-animator.json
(the one deployed on Vast — 55 nodes, 68KB).

Что делает этот base workflow:
  - LoadImage → ImageResizeKJv2 → [ClipVision + ref_images в WanVideoAnimateEmbeds]
  - VHS_LoadVideo → PoseAndFaceDetection → [face_images + pose_data]
  - pose_data → TSPoseDataSmoother → DrawViTPose → pose_images
  - CLIPText positive/negative → TextEmbedBridge → WanVideoSampler
  - WanVideoModelLoader → LoRAs → Sampler → Decode → Resize → VHS_VideoCombine
  НО: bg_images и mask в WanVideoAnimateEmbeds НЕ подключены → машина искажается.

Что добавляет этот builder:
  1. SAM2 + GrowMaskWithBlur + BlockifyMask → WanVideoAnimateEmbeds.mask
     (маска персонажа per-frame из ref-video)
  2. ImageBatchRepeatInterleaving: reference photo × num_frames → bg_images
     (frozen background — машина попиксельно из исходника)

Что правит:
  3. WanVideoModelLoader.base_precision: fp16 → bf16 (фикс precision mismatch)
  4. WanVideoAnimateEmbeds: face_strength 0.6 → 0.8 (лицо держится жёстче)
  5. Резолвит все Set/GetNode → прямые ссылки (API-safe, чистый граф)
  6. Удаляет preview vitpose VHS_VideoCombine#505 (не нужен в продакшене)
  7. Переписывает positive prompt под кейс (девушка+машина+пригород США)

Что сохраняется из оригинала:
  - TSPoseDataSmoother (cглаживание pose — плюс качеству)
  - Negative prompt (китайский проверенный список)
  - WanVideoSampler steps=4 (Lightx2v)
  - 4 LoRA стек в WanVideoLoraSelectMulti (проверенные)
  - Финальный ImageResizeKJv2 до 720×1280
"""

import json
import copy
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / 'workflows' / 'ofmhub-animator.json'
OUT_DIR = BASE / 'workflows' / 'custom'
OUT_UI = OUT_DIR / 'custom_v2_car_dance.json'

# === Параметры под нашу задачу ===
TARGET_FRAMES = 269      # 9 сек @ 30fps, кратно 4+1
FRAME_WINDOW = 81        # sliding window (дефолт OFMTech)
POSE_STRENGTH = 0.7
FACE_STRENGTH = 0.8      # 0.6 → 0.8 (лицо нашей девушки — жёстче держим)
OUTPUT_W = 720           # финальный resize (ImageResizeKJv2#158)
OUTPUT_H = 1280

POSITIVE_PROMPT = (
    "young woman in grey hoodie with hood up and black cap, black sweatpants, "
    "dancing in front of a parked car in suburban American driveway, "
    "natural daylight, detailed brick houses in background, "
    "static locked-off camera on tripod, completely still shot, "
    "no camera movement, no pan, no zoom, photorealistic, crisp details"
)

NODES_TO_REMOVE = {
    505,   # VHS_VideoCombine vitpose preview — не нужен
    534, 535, 536,  # Label (rgthree) — чисто декор
    501,   # Note — tooltip, не нужен в API
}


def load_wf():
    return json.load(open(SRC))


def resolve_set_get(wf):
    """Заменяет все SetNode/GetNode прямыми ссылками. Также обрабатывает случай
    когда SetNode output читается напрямую через link без GetNode (pass-through)."""
    nodes = {n['id']: n for n in wf['nodes']}
    links = wf['links']
    link_map = {L[0]: L for L in links}

    # Для каждого SetNode — найти реальный источник (что подано на его input[0])
    set_source = {}   # set_node_id → (real_src_node_id, real_src_slot, type)
    set_var = {}      # set_node_id → var_name
    for n in wf['nodes']:
        if n['type'] != 'SetNode':
            continue
        set_var[n['id']] = (n.get('widgets_values') or [''])[0]
        for inp in n.get('inputs', []) or []:
            if inp.get('link'):
                L = link_map.get(inp['link'])
                if L:
                    set_source[n['id']] = (L[1], L[2], L[5])
                break

    # GetNode id → var
    get_var = {}
    for n in wf['nodes']:
        if n['type'] == 'GetNode':
            get_var[n['id']] = (n.get('widgets_values') or [''])[0]

    # var → set_node_id (first one)
    var_to_set = {}
    for sn_id, var in set_var.items():
        if var not in var_to_set:
            var_to_set[var] = sn_id

    # Пересобрать links: src = SetNode | GetNode → заменяем на реальный источник
    new_links = []
    for L in links:
        lid, s, ss, dst, ds, t = L
        # src = GetNode → reroute to SetNode's source
        if s in get_var:
            var = get_var[s]
            if var not in var_to_set:
                continue
            sn_id = var_to_set[var]
            if sn_id not in set_source:
                continue
            real_src, real_slot, real_type = set_source[sn_id]
            new_links.append([lid, real_src, real_slot, dst, ds, real_type])
        # src = SetNode → reroute to its source
        elif s in set_source:
            real_src, real_slot, real_type = set_source[s]
            new_links.append([lid, real_src, real_slot, dst, ds, real_type])
        # dst = SetNode → drop (больше не нужен, заменили через source)
        elif dst in set_source:
            continue
        else:
            new_links.append(L)
    wf['links'] = new_links

    # Удалить Set/GetNode из nodes
    wf['nodes'] = [n for n in wf['nodes'] if n['type'] not in ('SetNode', 'GetNode')]


def prune_nodes(wf, remove_ids):
    wf['nodes'] = [n for n in wf['nodes'] if n['id'] not in remove_ids]
    wf['links'] = [L for L in wf['links'] if L[1] not in remove_ids and L[3] not in remove_ids]


def clean_orphan_links(wf):
    node_ids = {n['id'] for n in wf['nodes']}
    wf['links'] = [L for L in wf['links'] if L[1] in node_ids and L[3] in node_ids]
    link_ids = {L[0] for L in wf['links']}
    for n in wf['nodes']:
        for inp in n.get('inputs', []) or []:
            if inp.get('link') and inp['link'] not in link_ids:
                inp['link'] = None
        for out in n.get('outputs', []) or []:
            if out.get('links'):
                out['links'] = [l for l in out['links'] if l in link_ids]


def find_node(wf, nid):
    return next((n for n in wf['nodes'] if n['id'] == nid), None)


def next_node_id(wf):
    return max((n['id'] for n in wf['nodes']), default=0) + 1


def next_link_id(wf):
    return max((L[0] for L in wf['links']), default=0) + 1


def add_link(wf, src_id, src_slot, dst_id, dst_slot, link_type):
    lid = next_link_id(wf)
    wf['links'].append([lid, src_id, src_slot, dst_id, dst_slot, link_type])
    src = find_node(wf, src_id)
    dst = find_node(wf, dst_id)
    if src and src.get('outputs') and src_slot < len(src['outputs']):
        src['outputs'][src_slot].setdefault('links', []).append(lid)
    if dst and dst.get('inputs') and dst_slot < len(dst['inputs']):
        dst['inputs'][dst_slot]['link'] = lid
    return lid


def make_node(nid, ntype, pos, inputs, outputs, widgets=None, properties=None):
    return {
        'id': nid, 'type': ntype, 'pos': pos, 'size': [250, 100],
        'flags': {}, 'order': 0, 'mode': 0,
        'inputs': inputs, 'outputs': outputs,
        'properties': properties or {'Node name for S&R': ntype},
        'widgets_values': widgets or [],
    }


def add_sam2_chain(wf):
    """Добавляем SAM2 → GrowMaskWithBlur → BlockifyMask → WanVideoAnimateEmbeds.mask.
    image для SAM2 = тот же что у PoseAndFaceDetection (ref-video frames).
    coordinates_positive тоже от PoseAndFaceDetection (как в OFMTech)."""

    pose_node = find_node(wf, 89)      # PoseAndFaceDetection
    embeds = find_node(wf, 270)        # WanVideoAnimateEmbeds
    if not pose_node or not embeds:
        print("WARN: PoseAndFaceDetection#89 or Embeds#270 not found")
        return

    # Найти link, который подаётся на pose_node.images → чтоб взять тот же source
    pose_image_src = None
    for inp in pose_node.get('inputs') or []:
        if inp.get('name') == 'images' and inp.get('link'):
            L = next((l for l in wf['links'] if l[0] == inp['link']), None)
            if L:
                pose_image_src = (L[1], L[2])   # (src_node, src_slot)
            break

    if not pose_image_src:
        print("WARN: не нашёл source pose images")
        return

    # === Node: DownloadAndLoadSAM2Model ===
    sam_loader_id = next_node_id(wf)
    sam_loader = make_node(
        sam_loader_id, 'DownloadAndLoadSAM2Model',
        pos=[-1400, -200],
        inputs=[],
        outputs=[{'name': 'sam2_model', 'type': 'SAM2MODEL', 'links': []}],
        widgets=['sam2_hiera_large.safetensors', 'fp16', 'cuda', 'fp32'],
    )
    wf['nodes'].append(sam_loader)

    # === Node: Sam2Segmentation ===
    sam2_seg_id = next_node_id(wf)
    sam2_seg = make_node(
        sam2_seg_id, 'Sam2Segmentation',
        pos=[-1100, -200],
        inputs=[
            {'name': 'sam2_model', 'type': 'SAM2MODEL', 'link': None},
            {'name': 'image',      'type': 'IMAGE',     'link': None},
            {'name': 'coordinates_positive', 'type': 'STRING', 'link': None, 'shape': 7},
            {'name': 'coordinates_negative', 'type': 'STRING', 'link': None, 'shape': 7},
            {'name': 'bboxes',     'type': 'BBOX',      'link': None, 'shape': 7},
            {'name': 'mask',       'type': 'MASK',      'link': None, 'shape': 7},
        ],
        outputs=[{'name': 'mask', 'type': 'MASK', 'links': []}],
        widgets=[True, False],   # keep_model_loaded, individual_objects
    )
    wf['nodes'].append(sam2_seg)

    # === Node: GrowMaskWithBlur ===
    grow_id = next_node_id(wf)
    grow = make_node(
        grow_id, 'GrowMaskWithBlur',
        pos=[-800, -200],
        inputs=[{'name': 'mask', 'type': 'MASK', 'link': None}],
        outputs=[
            {'name': 'mask',          'type': 'MASK', 'links': []},
            {'name': 'mask_inverted', 'type': 'MASK', 'links': []},
        ],
        widgets=[6, 0, True, False, 10.0, 1.0, 1.0, False],
        # [expand, incremental_expandrate, tapered_corners, flip_input, blur_radius, lerp_alpha, decay_factor, fill_holes]
    )
    wf['nodes'].append(grow)

    # === Node: BlockifyMask ===
    blockify_id = next_node_id(wf)
    blockify = make_node(
        blockify_id, 'BlockifyMask',
        pos=[-500, -200],
        inputs=[{'name': 'masks', 'type': 'MASK', 'link': None}],
        outputs=[{'name': 'mask', 'type': 'MASK', 'links': []}],
        widgets=[8, 0.5],    # block_size, threshold
    )
    wf['nodes'].append(blockify)

    # Wiring:
    # SAM2Loader → SAM2Segmentation.sam2_model
    add_link(wf, sam_loader_id, 0, sam2_seg_id, 0, 'SAM2MODEL')
    # ref-video (тот же source что у pose) → SAM2Segmentation.image
    add_link(wf, pose_image_src[0], pose_image_src[1], sam2_seg_id, 1, 'IMAGE')

    # PoseAndFaceDetection.pose_data → SAM2Segmentation.coordinates_positive
    # (в оригинале OFMTech это делали — pose keypoints подаются как positive coords)
    # slot pose_data в PoseAndFaceDetection — output index 1 (после face_images)
    pose_data_slot = None
    for i, out in enumerate(pose_node.get('outputs') or []):
        if 'pose_data' in out.get('name', '').lower() or out.get('type') == 'POSE_DATA':
            pose_data_slot = i
            break
    if pose_data_slot is not None:
        add_link(wf, 89, pose_data_slot, sam2_seg_id, 2, 'STRING')

    # SAM2Segmentation → GrowMaskWithBlur
    add_link(wf, sam2_seg_id, 0, grow_id, 0, 'MASK')
    # GrowMaskWithBlur → BlockifyMask
    add_link(wf, grow_id, 0, blockify_id, 0, 'MASK')
    # BlockifyMask → WanVideoAnimateEmbeds.mask
    mask_slot = None
    for i, inp in enumerate(embeds.get('inputs') or []):
        if inp.get('name') == 'mask':
            mask_slot = i
            break
    if mask_slot is not None:
        add_link(wf, blockify_id, 0, 270, mask_slot, 'MASK')

    return {
        'sam_loader': sam_loader_id,
        'sam2_seg': sam2_seg_id,
        'grow': grow_id,
        'blockify': blockify_id,
    }


def add_bg_repeat(wf):
    """Добавляет ImageBatchRepeatInterleaving между ImageResizeKJv2#68 (reference photo)
    и WanVideoAnimateEmbeds#270.bg_images. repeats приходит из VHS.frame_count."""

    resize_photo = find_node(wf, 68)   # ImageResizeKJv2 (resize reference photo)
    embeds = find_node(wf, 270)
    vhs = find_node(wf, 75)            # VHS_LoadVideo
    if not (resize_photo and embeds and vhs):
        return None

    # frame_count slot в VHS_LoadVideo outputs (обычно INT slot)
    fc_slot = None
    for i, out in enumerate(vhs.get('outputs') or []):
        if out.get('type') == 'INT':
            fc_slot = i
            break
    if fc_slot is None:
        print("WARN: не нашёл frame_count output у VHS_LoadVideo")
        return None

    repeat_id = next_node_id(wf)
    repeat = make_node(
        repeat_id, 'ImageBatchRepeatInterleaving',
        pos=[-200, 200],
        inputs=[
            {'name': 'images',  'type': 'IMAGE', 'link': None},
            {'name': 'repeats', 'type': 'INT',   'link': None, 'widget': {'name': 'repeats'}},
            {'name': 'mask',    'type': 'MASK',  'link': None, 'shape': 7},
        ],
        outputs=[
            {'name': 'IMAGE', 'type': 'IMAGE', 'links': [], 'slot_index': 0},
            {'name': 'MASK',  'type': 'MASK',  'links': [], 'slot_index': 1},
        ],
        widgets=[TARGET_FRAMES],
    )
    wf['nodes'].append(repeat)

    # Wire: ImageResizeKJv2#68.IMAGE → repeat.images
    add_link(wf, 68, 0, repeat_id, 0, 'IMAGE')
    # VHS#75.frame_count → repeat.repeats
    add_link(wf, 75, fc_slot, repeat_id, 1, 'INT')
    # repeat.IMAGE → Embeds#270.bg_images
    bg_slot = None
    for i, inp in enumerate(embeds.get('inputs') or []):
        if inp.get('name') == 'bg_images':
            bg_slot = i
            break
    if bg_slot is not None:
        add_link(wf, repeat_id, 0, 270, bg_slot, 'IMAGE')

    return repeat_id


def apply_params(wf):
    # WanVideoModelLoader#22 widgets: fp16 → bf16 (фикс precision)
    n = find_node(wf, 22)
    if n:
        wv = n.get('widgets_values', [])
        if len(wv) >= 2:
            wv[1] = 'bf16'   # base_precision

    # WanVideoAnimateEmbeds#270 widgets:
    # [width, height, num_frames, force_offload, frame_window_size, colormatch,
    #  pose_strength, face_strength, tiled_vae]
    n = find_node(wf, 270)
    if n:
        wv = n.get('widgets_values', [])
        if len(wv) >= 9:
            wv[2] = TARGET_FRAMES        # num_frames: 81 → 269
            wv[4] = FRAME_WINDOW
            wv[6] = POSE_STRENGTH
            wv[7] = FACE_STRENGTH        # 0.6 → 0.8

    # Финальный ImageResizeKJv2#158 — 720×1280 (уже так, подтверждаем)
    n = find_node(wf, 158)
    if n:
        wv = n.get('widgets_values', [])
        if len(wv) >= 2:
            wv[0] = OUTPUT_W
            wv[1] = OUTPUT_H

    # Финальный VHS_VideoCombine#319 — добавить prefix
    n = find_node(wf, 319)
    if n and isinstance(n.get('widgets_values'), dict):
        n['widgets_values']['filename_prefix'] = 'video/car_dance'
        n['widgets_values']['frame_rate'] = 30

    # CLIPTextEncode — positive prompt (node id 493)
    n = find_node(wf, 493)
    if n:
        wv = n.get('widgets_values', [])
        if wv:
            wv[0] = POSITIVE_PROMPT


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wf = load_wf()
    print(f"Loaded {SRC.name}: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    resolve_set_get(wf)
    print(f"After Set/GetNode resolve: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    prune_nodes(wf, NODES_TO_REMOVE)
    clean_orphan_links(wf)
    print(f"After pruning: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    sam_ids = add_sam2_chain(wf)
    print(f"Added SAM2 chain: {sam_ids}")

    repeat_id = add_bg_repeat(wf)
    print(f"Added ImageBatchRepeatInterleaving: #{repeat_id}")

    apply_params(wf)

    json.dump(wf, open(OUT_UI, 'w'), indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_UI}")
    print(f"Size: {OUT_UI.stat().st_size / 1024:.1f} KB")

    from collections import Counter
    types = Counter(n['type'] for n in wf['nodes'])
    print(f"\nFinal: {len(wf['nodes'])} nodes, {len(wf['links'])} links")
    for t, c in types.most_common():
        print(f"  {c:2} × {t}")


if __name__ == '__main__':
    main()
