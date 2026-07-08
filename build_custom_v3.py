"""
Custom v3 Car Dance — apply research findings on top of v2.

Same source as v2: workflows/ofmhub-animator.json
(ofmhub-animator.json, 55 nodes, deployed on Vast).

Что применяем по сравнению с v2:

1. Удаляем TSPoseDataSmoother — не существует в официальных kijai репозиториях
   (search GitHub "TSPoseDataSmoother" = 0 результатов, kijai repos тоже нет).
   pose_data идёт напрямую в DrawViTPose (без smoothing).

2. Обновляем LoRA стек до 6 LoRA (community recommended):
   - lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16      @ 1.0  (оставляем rank256)
   - wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise            @ 1.0  (было 0.3, Kijai explicit rec)
   - wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise             @ 1.0  (ДОБАВЛЕНО parный)
   - Wan21_PusaV1_LoRA_14B_rank512_bf16                       @ 1.0  (было 0.9)
   - Wan2.2-Fun-A14B-InP-low-noise-HPS2.1                     @ 0.5  (оставляем)
   - WanAnimate_relight_lora_fp16                             @ 1.0  (ДОБАВЛЕНО — critical для integration)

3. WanVideoAnimateEmbeds параметры:
   - pose_strength 0.7 → 0.9 (жёсткий character lock)
   - face_strength 0.8 → 0.7 (community: ниже = меньше артефактов на лице)
   - colormatch 'disabled' → 'mkl' (спасает от color drift на 269 кадров)

4. WanVideoSampler:
   - steps 4 → 6
   - scheduler 'dpm++_sde' → 'unipc' (lightx2v explicit warning против dpm++_sde)
   - rope_function 'default' → 'comfy' (Blackwell optimize)
   - riflex_freq_index 0 → 6 (extends frame range)

5. GrowMaskWithBlur: expand 6→10, blur 10→14 (hoodie + hair edges)
6. BlockifyMask: block_size 8→16 (match temporal patch Wan)

7. ДОБАВЛЕНО: WanVideoContextOptions (context_frames=81, stride=4, overlap=24, schedule='uniform_standard')
   → WanVideoSampler.context_options — без этого color drift на 80+ кадрах

8. ДОБАВЛЕНО: Zeroed subject region для bg_images (по paper Wan-Animate):
   EmptyImage (W, H, batch=N, black) + ImageCompositeMasked(dest=ref×N, source=black, mask=subject)
   → bg_images с обнулённой subject area (как требует Wan-Animate replacement mode)

9. ДОБАВЛЕНО: Post-composite paste-back для 100% product preservation:
   ImageCompositeMasked(dest=ref×N, source=decoded_video, mask=subject_mask_grown)
   → финал где background попиксельно = reference photo, subject = generated video
   Это закрывает ~5% VAE encode/decode drift на неизменённом регионе.

Сохраняется: китайский negative prompt, TextEmbedBridge, базовая структура graph.
"""

import json
import copy
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / 'workflows' / 'ofmhub-animator.json'
OUT_DIR = BASE / 'workflows' / 'custom'
OUT_UI = OUT_DIR / 'custom_v3_car_dance.json'

# === Параметры (research-driven) ===
TARGET_FRAMES  = 269
FRAME_WINDOW   = 81
POSE_STRENGTH  = 0.9        # 0.7 → 0.9 (research: жёсткий pose lock)
FACE_STRENGTH  = 0.7        # 0.8 → 0.7 (research: community снижает)
COLORMATCH     = 'mkl'      # 'disabled' → 'mkl' (предотвращает color drift)
SAMPLER_STEPS  = 6          # 4 → 6
SAMPLER_SCHED  = 'unipc'    # 'dpm++_sde' → 'unipc'
ROPE_FUNCTION  = 'comfy'    # 'default' → 'comfy' (Blackwell)
RIFLEX_FREQ    = 6          # 0 → 6 (extends range)
CTX_FRAMES     = 81
CTX_STRIDE     = 4
CTX_OVERLAP    = 24         # 16 → 24 (меньше color drift)
CTX_SCHEDULE   = 'uniform_standard'
GROW_EXPAND    = 10         # 6 → 10
GROW_BLUR      = 14.0       # 10 → 14
BLOCKIFY_SIZE  = 16         # 8 → 16
OUTPUT_W       = 720
OUTPUT_H       = 1280

POSITIVE_PROMPT = (
    "young woman in grey hoodie with hood up, black cap underneath, "
    "black loose sweatpants, cream colored sneakers, "
    "dancing confidently in front of a parked car in suburban American driveway, "
    "bright natural daylight, brick houses with trees in background, "
    "static locked-off camera on tripod, photorealistic, sharp details"
)

# LoRA стек (Multi) — 6 LoRA в widgets_values (pairs name+strength)
LORA_STACK = [
    'lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors',  1.0,
    'wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors',        1.0,
    'wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors',         1.0,
    'Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors',                   1.0,
    'Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors',                 0.5,
    'WanAnimate_relight_lora_fp16.safetensors',                         1.0,
]

NODES_TO_REMOVE = {
    532,        # TSPoseDataSmoother — не существует в официальных kijai nodes
    505,        # preview vitpose VHS
    534, 535, 536, 501,  # Labels + Note
}


# ============================================================
# Utility (общие с v2)
# ============================================================

def load_wf():
    return json.load(open(SRC))


def resolve_set_get(wf):
    nodes = {n['id']: n for n in wf['nodes']}
    links = wf['links']
    link_map = {L[0]: L for L in links}

    set_source, set_var, var_to_set = {}, {}, {}
    for n in wf['nodes']:
        if n['type'] != 'SetNode': continue
        var = (n.get('widgets_values') or [''])[0]
        set_var[n['id']] = var
        for inp in n.get('inputs') or []:
            if inp.get('link'):
                L = link_map.get(inp['link'])
                if L: set_source[n['id']] = (L[1], L[2], L[5])
                break
    for sn_id, var in set_var.items():
        if var not in var_to_set: var_to_set[var] = sn_id

    get_var = {n['id']: (n.get('widgets_values') or [''])[0]
               for n in wf['nodes'] if n['type'] == 'GetNode'}

    new_links = []
    for L in links:
        lid, s, ss, dst, ds, t = L
        if s in get_var:
            var = get_var[s]
            sn_id = var_to_set.get(var)
            if sn_id and sn_id in set_source:
                rs, rss, rt = set_source[sn_id]
                new_links.append([lid, rs, rss, dst, ds, rt])
        elif s in set_source:
            rs, rss, rt = set_source[s]
            new_links.append([lid, rs, rss, dst, ds, rt])
        elif dst in set_source:
            continue
        else:
            new_links.append(L)
    wf['links'] = new_links
    wf['nodes'] = [n for n in wf['nodes'] if n['type'] not in ('SetNode', 'GetNode')]


def prune(wf, ids):
    wf['nodes'] = [n for n in wf['nodes'] if n['id'] not in ids]
    wf['links'] = [L for L in wf['links'] if L[1] not in ids and L[3] not in ids]


def clean_orphans(wf):
    node_ids = {n['id'] for n in wf['nodes']}
    wf['links'] = [L for L in wf['links'] if L[1] in node_ids and L[3] in node_ids]
    link_ids = {L[0] for L in wf['links']}
    for n in wf['nodes']:
        for inp in n.get('inputs') or []:
            if inp.get('link') and inp['link'] not in link_ids:
                inp['link'] = None
        for out in n.get('outputs') or []:
            if out.get('links'):
                out['links'] = [l for l in out['links'] if l in link_ids]


def find_node(wf, nid):
    return next((n for n in wf['nodes'] if n['id'] == nid), None)


def next_nid(wf):
    return max((n['id'] for n in wf['nodes']), default=0) + 1


def next_lid(wf):
    return max((L[0] for L in wf['links']), default=0) + 1


def add_link(wf, s, ss, d, ds, t):
    lid = next_lid(wf)
    wf['links'].append([lid, s, ss, d, ds, t])
    sn = find_node(wf, s)
    dn = find_node(wf, d)
    if sn and sn.get('outputs') and ss < len(sn['outputs']):
        sn['outputs'][ss].setdefault('links', []).append(lid)
    if dn and dn.get('inputs') and ds < len(dn['inputs']):
        dn['inputs'][ds]['link'] = lid
    return lid


def make_node(nid, ntype, pos, inputs, outputs, widgets=None):
    return {
        'id': nid, 'type': ntype, 'pos': pos, 'size': [250, 100],
        'flags': {}, 'order': 0, 'mode': 0,
        'inputs': inputs, 'outputs': outputs,
        'properties': {'Node name for S&R': ntype},
        'widgets_values': widgets or [],
    }


def replace_input_link(wf, dst_id, input_name, new_src_id, new_src_slot, link_type):
    """Удалить текущий link идущий в dst.input_name, создать новый от new_src."""
    dst = find_node(wf, dst_id)
    if not dst: return
    inp = next((i for i in dst.get('inputs') or [] if i.get('name') == input_name), None)
    if not inp: return
    # Удалить старый link
    old_lid = inp.get('link')
    if old_lid:
        wf['links'] = [L for L in wf['links'] if L[0] != old_lid]
    # Новый
    add_link(wf, new_src_id, new_src_slot, dst_id, next(i for i, x in enumerate(dst['inputs']) if x.get('name') == input_name), link_type)


def find_slot(node, slot_name, is_input=True):
    seq = node.get('inputs' if is_input else 'outputs') or []
    for i, s in enumerate(seq):
        if s.get('name') == slot_name:
            return i
    return None


# ============================================================
# Блоки сборки v3
# ============================================================

def rewire_pose_without_smoother(wf):
    """TSPoseDataSmoother#532 удаляется. Ранее: Pose#89 → Smoother#532 → DrawViTPose#533.
    Теперь: Pose#89.pose_data → DrawViTPose#533.pose_data напрямую.
    Также WanVideoAnimateEmbeds#270.pose_images сейчас идёт от Smoother, переключим на DrawViTPose#533."""
    pose = find_node(wf, 89)
    drawpose = find_node(wf, 533)
    embeds = find_node(wf, 270)
    if not (pose and drawpose and embeds):
        return

    pose_data_slot = find_slot(pose, 'pose_data', is_input=False)
    if pose_data_slot is None:
        for i, o in enumerate(pose.get('outputs') or []):
            if o.get('type') == 'POSE_KEYPOINT' or 'pose' in (o.get('name') or '').lower():
                pose_data_slot = i; break

    drawpose_input_slot = find_slot(drawpose, 'pose_data', is_input=True)
    embeds_pose_slot    = find_slot(embeds,   'pose_images', is_input=True)
    drawpose_img_out    = find_slot(drawpose, 'IMAGE', is_input=False)
    if drawpose_img_out is None:
        drawpose_img_out = 0

    # Сначала удалим все links ведущие в DrawViTPose.pose_data и Embeds.pose_images
    for (dst_id, dst_slot) in [(533, drawpose_input_slot), (270, embeds_pose_slot)]:
        if dst_slot is None: continue
        wf['links'] = [L for L in wf['links'] if not (L[3] == dst_id and L[4] == dst_slot)]
        dn = find_node(wf, dst_id)
        if dn: dn['inputs'][dst_slot]['link'] = None

    # Добавить: Pose#89.pose_data → DrawViTPose#533.pose_data
    if pose_data_slot is not None and drawpose_input_slot is not None:
        add_link(wf, 89, pose_data_slot, 533, drawpose_input_slot, 'POSE_KEYPOINT')

    # Добавить: DrawViTPose#533.IMAGE → Embeds#270.pose_images
    if embeds_pose_slot is not None:
        add_link(wf, 533, drawpose_img_out, 270, embeds_pose_slot, 'IMAGE')


def add_sam2_chain(wf):
    """SAM2 → GrowMaskWithBlur → BlockifyMask → Embeds.mask.
    image source = тот же что у PoseAndFaceDetection#89.images."""
    pose = find_node(wf, 89)
    embeds = find_node(wf, 270)
    if not pose or not embeds: return None

    pose_image_src = None
    for inp in pose.get('inputs') or []:
        if inp.get('name') == 'images' and inp.get('link'):
            L = next((l for l in wf['links'] if l[0] == inp['link']), None)
            if L: pose_image_src = (L[1], L[2]); break
    if not pose_image_src: return None

    sam_loader = make_node(next_nid(wf), 'DownloadAndLoadSAM2Model', [-1600, -200],
        inputs=[],
        outputs=[{'name': 'sam2_model', 'type': 'SAM2MODEL', 'links': []}],
        widgets=['sam2_hiera_large.safetensors', 'fp16', 'cuda', 'fp32'])
    wf['nodes'].append(sam_loader)
    sam_loader_id = sam_loader['id']

    sam2_seg = make_node(next_nid(wf), 'Sam2Segmentation', [-1300, -200],
        inputs=[
            {'name': 'sam2_model', 'type': 'SAM2MODEL', 'link': None},
            {'name': 'image',      'type': 'IMAGE',     'link': None},
            {'name': 'coordinates_positive', 'type': 'STRING', 'link': None, 'shape': 7},
            {'name': 'coordinates_negative', 'type': 'STRING', 'link': None, 'shape': 7},
            {'name': 'bboxes',     'type': 'BBOX',      'link': None, 'shape': 7},
            {'name': 'mask',       'type': 'MASK',      'link': None, 'shape': 7},
        ],
        outputs=[{'name': 'mask', 'type': 'MASK', 'links': []}],
        widgets=[True, False])
    wf['nodes'].append(sam2_seg)
    sam2_seg_id = sam2_seg['id']

    grow = make_node(next_nid(wf), 'GrowMaskWithBlur', [-1000, -200],
        inputs=[{'name': 'mask', 'type': 'MASK', 'link': None}],
        outputs=[
            {'name': 'mask',          'type': 'MASK', 'links': []},
            {'name': 'mask_inverted', 'type': 'MASK', 'links': []},
        ],
        widgets=[GROW_EXPAND, 0, True, False, GROW_BLUR, 1.0, 1.0, False])
    wf['nodes'].append(grow)
    grow_id = grow['id']

    blockify = make_node(next_nid(wf), 'BlockifyMask', [-700, -200],
        inputs=[{'name': 'masks', 'type': 'MASK', 'link': None}],
        outputs=[{'name': 'mask', 'type': 'MASK', 'links': []}],
        widgets=[BLOCKIFY_SIZE, 0.5])
    wf['nodes'].append(blockify)
    blockify_id = blockify['id']

    # Wire
    add_link(wf, sam_loader_id, 0, sam2_seg_id, 0, 'SAM2MODEL')
    add_link(wf, pose_image_src[0], pose_image_src[1], sam2_seg_id, 1, 'IMAGE')
    pose_data_slot = find_slot(pose, 'pose_data', is_input=False)
    if pose_data_slot is None: pose_data_slot = 1
    add_link(wf, 89, pose_data_slot, sam2_seg_id, 2, 'STRING')
    add_link(wf, sam2_seg_id, 0, grow_id, 0, 'MASK')
    add_link(wf, grow_id, 0, blockify_id, 0, 'MASK')

    mask_slot = find_slot(embeds, 'mask', is_input=True)
    add_link(wf, blockify_id, 0, 270, mask_slot, 'MASK')

    return {'sam_loader': sam_loader_id, 'sam2_seg': sam2_seg_id,
            'grow': grow_id, 'blockify': blockify_id}


def add_zeroed_bg(wf):
    """Block D: zeroed subject region for bg_images (Wan-Animate paper spec).

    ref_×N = ImageBatchRepeatInterleaving(resize#68, repeats=VHS#75.frame_count)
    black_×N = EmptyImage(W, H, batch=N, color=0)
    zeroed_bg = ImageCompositeMasked(dest=ref_×N, source=black_×N, mask=grown_mask)
    zeroed_bg → Embeds#270.bg_images"""

    embeds = find_node(wf, 270)
    vhs    = find_node(wf, 75)
    resize = find_node(wf, 68)
    gs     = find_node(wf, 309)   # GetImageSizeAndCount (W, H, count)
    if not all((embeds, vhs, resize, gs)): return None

    # Find grown_mask node (GrowMaskWithBlur) — ищем по type (только один)
    grow = next((n for n in wf['nodes'] if n['type'] == 'GrowMaskWithBlur'), None)
    if not grow: return None

    fc_slot = next((i for i, o in enumerate(vhs.get('outputs') or [])
                    if o.get('type') == 'INT'), None)
    if fc_slot is None: return None

    # GetImageSizeAndCount outputs: [width, height, count] — типично
    gs_w_slot = next((i for i, o in enumerate(gs.get('outputs') or [])
                      if o.get('type') == 'INT' and (o.get('name') or '').lower().startswith('w')), None)
    gs_h_slot = next((i for i, o in enumerate(gs.get('outputs') or [])
                      if o.get('type') == 'INT' and (o.get('name') or '').lower().startswith('h')), None)
    if gs_w_slot is None: gs_w_slot = 1  # fallback (width обычно 1-й)
    if gs_h_slot is None: gs_h_slot = 2

    # 1. ImageBatchRepeatInterleaving
    repeat = make_node(next_nid(wf), 'ImageBatchRepeatInterleaving', [-200, 200],
        inputs=[
            {'name': 'images',  'type': 'IMAGE', 'link': None},
            {'name': 'repeats', 'type': 'INT',   'link': None, 'widget': {'name': 'repeats'}},
            {'name': 'mask',    'type': 'MASK',  'link': None, 'shape': 7},
        ],
        outputs=[
            {'name': 'IMAGE', 'type': 'IMAGE', 'links': [], 'slot_index': 0},
            {'name': 'MASK',  'type': 'MASK',  'links': [], 'slot_index': 1},
        ],
        widgets=[TARGET_FRAMES])
    wf['nodes'].append(repeat); repeat_id = repeat['id']
    add_link(wf, 68, 0, repeat_id, 0, 'IMAGE')
    add_link(wf, 75, fc_slot, repeat_id, 1, 'INT')

    # 2. EmptyImage (native) — generates black batch
    empty = make_node(next_nid(wf), 'EmptyImage', [-200, 350],
        inputs=[
            {'name': 'width',       'type': 'INT', 'link': None, 'widget': {'name': 'width'}},
            {'name': 'height',      'type': 'INT', 'link': None, 'widget': {'name': 'height'}},
            {'name': 'batch_size',  'type': 'INT', 'link': None, 'widget': {'name': 'batch_size'}},
            {'name': 'color',       'type': 'INT', 'link': None, 'widget': {'name': 'color'}},
        ],
        outputs=[{'name': 'IMAGE', 'type': 'IMAGE', 'links': []}],
        widgets=[512, 512, TARGET_FRAMES, 0])
    wf['nodes'].append(empty); empty_id = empty['id']
    add_link(wf, 309, gs_w_slot, empty_id, 0, 'INT')
    add_link(wf, 309, gs_h_slot, empty_id, 1, 'INT')
    add_link(wf, 75,  fc_slot,   empty_id, 2, 'INT')

    # 3. ImageCompositeMasked (native): destination=ref×N, source=black, mask=subject_mask
    compose = make_node(next_nid(wf), 'ImageCompositeMasked', [100, 200],
        inputs=[
            {'name': 'destination',   'type': 'IMAGE', 'link': None},
            {'name': 'source',        'type': 'IMAGE', 'link': None},
            {'name': 'x',             'type': 'INT',   'link': None, 'widget': {'name': 'x'}},
            {'name': 'y',             'type': 'INT',   'link': None, 'widget': {'name': 'y'}},
            {'name': 'resize_source', 'type': 'BOOLEAN','link': None, 'widget': {'name': 'resize_source'}},
            {'name': 'mask',          'type': 'MASK',  'link': None, 'shape': 7},
        ],
        outputs=[{'name': 'IMAGE', 'type': 'IMAGE', 'links': []}],
        widgets=[0, 0, False])
    wf['nodes'].append(compose); compose_id = compose['id']
    add_link(wf, repeat_id, 0, compose_id, 0, 'IMAGE')
    add_link(wf, empty_id,  0, compose_id, 1, 'IMAGE')
    add_link(wf, grow['id'], 0, compose_id, 5, 'MASK')  # GrowMask output slot 0 (mask)

    # 4. compose → Embeds.bg_images
    bg_slot = find_slot(embeds, 'bg_images', is_input=True)
    add_link(wf, compose_id, 0, 270, bg_slot, 'IMAGE')

    return {'repeat': repeat_id, 'empty': empty_id, 'compose_bg': compose_id}


def add_context_options(wf):
    """WanVideoContextOptions → WanVideoSampler.context_options."""
    sampler = find_node(wf, 273)
    if not sampler: return None

    ctx = make_node(next_nid(wf), 'WanVideoContextOptions', [200, 600],
        inputs=[],
        outputs=[{'name': 'context_options', 'type': 'WANVIDCONTEXT', 'links': []}],
        widgets=[CTX_SCHEDULE, CTX_FRAMES, CTX_STRIDE, CTX_OVERLAP, 'inner', True, False])
        # widgets order из nodes.py: [context_schedule, context_frames, context_stride,
        #                             context_overlap, fuse_method, verbose, looped]
    wf['nodes'].append(ctx); ctx_id = ctx['id']

    ctx_slot = find_slot(sampler, 'context_options', is_input=True)
    if ctx_slot is not None:
        add_link(wf, ctx_id, 0, 273, ctx_slot, 'WANVIDCONTEXT')
    return ctx_id


def add_post_composite(wf):
    """Block F (paste-back): ImageCompositeMasked(dest=ref×N, source=decoded, mask=grown_mask)
    → ImageResizeKJv2#158 → VHS_VideoCombine#319.

    Для дестинации reference×N переиспользуем node ImageBatchRepeatInterleaving
    который добавлен в add_zeroed_bg (его output 0 = ref×N)."""

    decode = find_node(wf, 28)
    resize_out = find_node(wf, 158)  # финальный ресайз (720×1280)
    grow = next((n for n in wf['nodes'] if n['type'] == 'GrowMaskWithBlur'), None)
    repeat = next((n for n in wf['nodes'] if n['type'] == 'ImageBatchRepeatInterleaving'), None)
    if not all((decode, resize_out, grow, repeat)): return None

    # Найти текущий link decode.IMAGE → resize_out.image и разорвать
    # (заменим на decode → composite → resize_out)
    # Comfy: Decode output "images" idx 0
    decode_out_slot = 0
    resize_in_slot = find_slot(resize_out, 'image', is_input=True)
    # Удалить существующий link идущий в resize_out.image
    if resize_in_slot is not None:
        old_lid = resize_out['inputs'][resize_in_slot].get('link')
        if old_lid:
            wf['links'] = [L for L in wf['links'] if L[0] != old_lid]
            resize_out['inputs'][resize_in_slot]['link'] = None

    # ImageCompositeMasked paste-back
    compose = make_node(next_nid(wf), 'ImageCompositeMasked', [600, 800],
        inputs=[
            {'name': 'destination',   'type': 'IMAGE', 'link': None},
            {'name': 'source',        'type': 'IMAGE', 'link': None},
            {'name': 'x',             'type': 'INT',   'link': None, 'widget': {'name': 'x'}},
            {'name': 'y',             'type': 'INT',   'link': None, 'widget': {'name': 'y'}},
            {'name': 'resize_source', 'type': 'BOOLEAN','link': None, 'widget': {'name': 'resize_source'}},
            {'name': 'mask',          'type': 'MASK',  'link': None, 'shape': 7},
        ],
        outputs=[{'name': 'IMAGE', 'type': 'IMAGE', 'links': []}],
        widgets=[0, 0, False])
    wf['nodes'].append(compose); compose_id = compose['id']

    # Wire
    add_link(wf, repeat['id'], 0, compose_id, 0, 'IMAGE')    # dest = ref×N
    add_link(wf, 28, decode_out_slot, compose_id, 1, 'IMAGE')  # source = decoded video
    add_link(wf, grow['id'], 0, compose_id, 5, 'MASK')       # mask = grown subject mask

    # composite.IMAGE → resize_out.image
    add_link(wf, compose_id, 0, 158, resize_in_slot, 'IMAGE')

    return compose_id


def apply_params(wf):
    # WanVideoModelLoader#22: bf16 + правильный порядок
    n = find_node(wf, 22)
    if n:
        wv = n.get('widgets_values', [])
        # [model, base_precision, quantization, load_device, attention_mode, rms_norm_function]
        if len(wv) >= 6:
            wv[1] = 'bf16'
            wv[2] = 'fp8_e4m3fn'
            wv[3] = 'offload_device'
            wv[4] = 'sdpa'
            wv[5] = 'default'

    # WanVideoAnimateEmbeds#270
    n = find_node(wf, 270)
    if n:
        wv = n.get('widgets_values', [])
        # [width, height, num_frames, force_offload, frame_window_size, colormatch,
        #  pose_strength, face_strength, tiled_vae]
        if len(wv) >= 9:
            wv[2] = TARGET_FRAMES
            wv[4] = FRAME_WINDOW
            wv[5] = COLORMATCH
            wv[6] = POSE_STRENGTH
            wv[7] = FACE_STRENGTH

    # WanVideoSampler#273
    n = find_node(wf, 273)
    if n:
        wv = n.get('widgets_values', [])
        # [steps, cfg, shift, seed, seed_control, force_offload, scheduler,
        #  riflex_freq_index, denoise_strength, seed_? / something, rope_function,
        #  start_step, end_step, add_noise_to_samples]
        if len(wv) >= 11:
            wv[0] = SAMPLER_STEPS
            wv[1] = 1
            wv[2] = 5
            wv[6] = SAMPLER_SCHED
            wv[7] = RIFLEX_FREQ
            wv[8] = 1.0           # denoise_strength полный (первый pass)
            wv[10] = ROPE_FUNCTION

    # WanVideoLoraSelectMulti#354 — 6 LoRA (исходно 5 слотов + 2 флага)
    n = find_node(wf, 354)
    if n:
        # Widgets layout: name1, str1, name2, str2, ..., name5, str5, low_mem_load, merge_loras
        # Для 6 LoRA нужно 12 значений + 2 флага = 14
        # Заменим widgets полностью
        wv = LORA_STACK + [False, False]
        n['widgets_values'] = wv

    # GrowMaskWithBlur — params (SAM chain нода, уже создана в add_sam2_chain с нашими значениями)

    # Финальный VHS_VideoCombine#319
    n = find_node(wf, 319)
    if n and isinstance(n.get('widgets_values'), dict):
        n['widgets_values']['filename_prefix'] = 'video/car_dance_v3'
        n['widgets_values']['frame_rate'] = 30

    # Финальный ImageResizeKJv2#158
    n = find_node(wf, 158)
    if n:
        wv = n.get('widgets_values', [])
        if len(wv) >= 2:
            wv[0] = OUTPUT_W
            wv[1] = OUTPUT_H

    # CLIPTextEncode#493 (positive prompt)
    n = find_node(wf, 493)
    if n and n.get('widgets_values'):
        n['widgets_values'][0] = POSITIVE_PROMPT


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wf = load_wf()
    print(f"Loaded: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    resolve_set_get(wf)
    print(f"Set/Get resolved: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    prune(wf, NODES_TO_REMOVE)
    clean_orphans(wf)
    print(f"Pruned (incl. TSPoseDataSmoother): {len(wf['nodes'])} nodes")

    rewire_pose_without_smoother(wf)

    sam_ids = add_sam2_chain(wf)
    print(f"SAM2 chain: {sam_ids}")

    bg_ids = add_zeroed_bg(wf)
    print(f"Zeroed bg chain: {bg_ids}")

    ctx_id = add_context_options(wf)
    print(f"Context options: #{ctx_id}")

    post_id = add_post_composite(wf)
    print(f"Post-composite: #{post_id}")

    apply_params(wf)

    json.dump(wf, open(OUT_UI, 'w'), indent=2, ensure_ascii=False)
    sz = OUT_UI.stat().st_size / 1024
    print(f"\n=== Saved v3: {OUT_UI}")
    print(f"Size: {sz:.1f} KB")

    from collections import Counter
    types = Counter(n['type'] for n in wf['nodes'])
    print(f"\nFinal: {len(wf['nodes'])} nodes, {len(wf['links'])} links")
    for t, c in types.most_common():
        print(f"  {c:2} × {t}")


if __name__ == '__main__':
    main()
