"""
Build custom_v1_car_dance workflow from OFMTech_BarbecueMotion.

Цель: «танцующая девушка на фоне машины, машина = продукт = идеально сохранён».

Что делает:
  1. Читает OFMTech_BarbecueMotion.json (UI-формат).
  2. Удаляет ветки которые нам не нужны:
       - 2-shot branch (WanVideoAnimateEmbeds #274 + Any Switch #276)
       - RIFE interpolation (#189, #212) + preview VHS (#211, #213)
       - Film grain (#190)
       - Preview видосы (#30, #75, #174, #181) — оставляем только финальный #214
       - ImageConcatMulti composites (#66, #77), showAnything (#314), mathFloat (#207),
         CRTFirstLastFrameSelector (#227), SeedVR2BlockSwap (#322),
         GetNode fps (#203-205)
  3. Резолвит все Set/GetNode → прямые ссылки (чистый граф, API-safe).
  4. Перекоммутирует wiring под кастомную схему:
       - reference photo → ref_images + bg_images (репликация × N)
       - SAM2 работает на reference photo (не на ref-video)
  5. Выставляет параметры под: 9сек @ 30fps = 269 кадров, 540×960.
  6. Сохраняет UI-копию (custom_v1_car_dance.json) и API-копию (.api.json).
"""

import json
import copy
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / 'workflows' / 'OFMTech_BarbecueMotion.json'
OUT_DIR = BASE / 'workflows' / 'custom'
OUT_UI = OUT_DIR / 'custom_v1_car_dance.json'

# === Настройки под нашу задачу ===
TARGET_WIDTH = 540
TARGET_HEIGHT = 960
TARGET_FRAMES = 269          # 9 сек @ 30fps, кратно 4 + 1
TARGET_FPS = 30
FRAME_WINDOW = 81            # sliding attention window (дефолт WanAnimate)
POSE_STRENGTH = 0.7
FACE_STRENGTH = 0.8          # поднят с 0.6 — лицо нашей девушки должно держаться

# Ноды к удалению (id из OFMTech карты)
NODES_TO_REMOVE = {
    # 2-shot duplicate branch
    274,   # WanVideoAnimateEmbeds #2
    276,   # Any Switch (rgthree)
    242,   # GetNode frame_count для window_size #2
    # RIFE + пост-интерполяция
    189,   # RIFE VFI
    212,   # RIFE VFI
    190,   # ProPostFilmGrain
    # Preview VHS (всё кроме финального #214)
    30,    # VHS_VideoCombine composite preview
    75,    # VHS_VideoCombine bg preview
    174,   # VHS_VideoCombine face preview
    181,   # VHS_VideoCombine pose preview
    211,   # VHS_VideoCombine RIFE preview
    213,   # VHS_VideoCombine RIFE preview
    # Preview composites / utility
    66,    # ImageConcatMulti
    77,    # ImageConcatMulti
    314,   # easy showAnything
    207,   # easy mathFloat
    227,   # CRTFirstLastFrameSelector
    322,   # SeedVR2BlockSwap
    203, 204, 205,  # GetNode fps
    # Preview-only DrawMaskOnImage (если bg берём с reference photo напрямую)
    316,   # DrawMaskOnImage — мы не используем его output для bg
    252,   # ShowText|pysssss
    315,   # Label
}


def load_workflow(path: Path) -> dict:
    return json.load(open(path))


def build_lookups(wf: dict):
    nodes = {n['id']: n for n in wf['nodes']}
    links = wf['links']
    link_map = {L[0]: L for L in links}
    return nodes, links, link_map


def resolve_set_get(wf: dict):
    """Замени все SetNode/GetNode прямыми ссылками.
    После этого можно удалить сами Set/Get ноды."""
    nodes, links, link_map = build_lookups(wf)

    # var_name -> (src_node_id, src_slot_idx, link_type)
    var_source = {}
    for n in wf['nodes']:
        if n['type'] != 'SetNode':
            continue
        var = (n.get('widgets_values') or [''])[0]
        # SetNode имеет один input (slot 0), куда приходит источник
        for inp in n.get('inputs', []) or []:
            lid = inp.get('link')
            if lid and lid in link_map:
                L = link_map[lid]
                # L = [lid, src_node, src_slot, dst_node, dst_slot, type]
                var_source[var] = (L[1], L[2], L[5])
                break

    # GetNode id -> var
    get_nodes = {}
    for n in wf['nodes']:
        if n['type'] == 'GetNode':
            var = (n.get('widgets_values') or [''])[0]
            get_nodes[n['id']] = var

    # Найти все links, где src = GetNode, и заменить их src на реального источника
    new_links = []
    max_link = max((L[0] for L in links), default=0)
    for L in links:
        lid, s, ss, dst, ds, t = L
        if s in get_nodes:
            var = get_nodes[s]
            if var not in var_source:
                # SetNode для этой переменной не найден — пропускаем link
                continue
            src_id, src_slot, src_type = var_source[var]
            new_links.append([lid, src_id, src_slot, dst, ds, src_type])
        else:
            new_links.append(L)
    wf['links'] = new_links

    # Удаляем все SetNode и GetNode из списка нод
    wf['nodes'] = [n for n in wf['nodes'] if n['type'] not in ('SetNode', 'GetNode')]


def prune_nodes(wf: dict, remove_ids: set):
    """Удалить ноды и все связанные с ними links."""
    wf['nodes'] = [n for n in wf['nodes'] if n['id'] not in remove_ids]
    wf['links'] = [L for L in wf['links']
                   if L[1] not in remove_ids and L[3] not in remove_ids]


def clean_orphan_input_links(wf: dict):
    """После удаления нод некоторые inputs ссылаются на несуществующие links. Зачищаем.
    Также выкидываем links которые ведут в/из удалённых нод."""
    node_ids = {n['id'] for n in wf['nodes']}
    # Выкидываем dangling links
    wf['links'] = [L for L in wf['links'] if L[1] in node_ids and L[3] in node_ids]
    link_ids = {L[0] for L in wf['links']}
    for n in wf['nodes']:
        for inp in n.get('inputs', []) or []:
            if inp.get('link') and inp['link'] not in link_ids:
                inp['link'] = None
        # outputs.links тоже
        for out in n.get('outputs', []) or []:
            if out.get('links'):
                out['links'] = [l for l in out['links'] if l in link_ids]


def next_node_id(wf):
    return max((n['id'] for n in wf['nodes']), default=0) + 1

def next_link_id(wf):
    return max((L[0] for L in wf['links']), default=0) + 1


def add_image_batch_repeat(wf):
    """Вставляет ImageBatchRepeatInterleaving между ImageResizeKJv2#64 (reference photo)
    и WanVideoAnimateEmbeds#62.bg_images.
    repeats = output frame_count из VHS_LoadVideo#63."""

    resize_64 = find_node_by_id(wf, 64)
    embeds_62 = find_node_by_id(wf, 62)
    vhs_63 = find_node_by_id(wf, 63)
    if not (resize_64 and embeds_62 and vhs_63):
        return None

    new_id = next_node_id(wf)

    # INPUT_TYPES: required images (IMAGE), repeats (INT); optional mask (MASK)
    # Выходов 2: IMAGE, MASK
    new_node = {
        'id': new_id,
        'type': 'ImageBatchRepeatInterleaving',
        'pos': [600, -200],
        'size': [240, 80],
        'flags': {},
        'order': 0,
        'mode': 0,
        'inputs': [
            {'name': 'images',  'type': 'IMAGE', 'link': None},
            {'name': 'repeats', 'type': 'INT',   'link': None, 'widget': {'name': 'repeats'}},
            {'name': 'mask',    'type': 'MASK',  'link': None, 'shape': 7},
        ],
        'outputs': [
            {'name': 'IMAGE', 'type': 'IMAGE', 'links': [], 'slot_index': 0},
            {'name': 'MASK',  'type': 'MASK',  'links': [], 'slot_index': 1},
        ],
        'properties': {'Node name for S&R': 'ImageBatchRepeatInterleaving'},
        'widgets_values': [TARGET_FRAMES],  # дефолт; реально придёт через link от VHS.frame_count
    }
    wf['nodes'].append(new_node)

    # Link A: ImageResizeKJv2#64.output[0] (IMAGE) → new.inputs[0] (images)
    lid_a = next_link_id(wf)
    wf['links'].append([lid_a, 64, 0, new_id, 0, 'IMAGE'])
    new_node['inputs'][0]['link'] = lid_a
    # добавить в outputs[0].links у #64
    if resize_64.get('outputs') and len(resize_64['outputs']) > 0:
        resize_64['outputs'][0].setdefault('links', []).append(lid_a)

    # Link B: VHS_LoadVideo#63.frame_count → new.inputs[1] (repeats, INT)
    # frame_count у VHS_LoadVideo обычно outputs[2] (IMAGE, audio, frame_count)
    # Найдём точно:
    vhs_frame_count_slot = None
    for i, out in enumerate(vhs_63.get('outputs') or []):
        if out.get('type') == 'INT' or out.get('name', '').lower() == 'frame_count':
            vhs_frame_count_slot = i
            break
    if vhs_frame_count_slot is not None:
        lid_b = next_link_id(wf)
        wf['links'].append([lid_b, 63, vhs_frame_count_slot, new_id, 1, 'INT'])
        new_node['inputs'][1]['link'] = lid_b
        vhs_63['outputs'][vhs_frame_count_slot].setdefault('links', []).append(lid_b)

    # Link C: new.outputs[0] (IMAGE) → WanVideoAnimateEmbeds#62.bg_images
    bg_input_slot = None
    for i, inp in enumerate(embeds_62.get('inputs') or []):
        if inp.get('name') == 'bg_images':
            bg_input_slot = i
            break
    if bg_input_slot is not None:
        lid_c = next_link_id(wf)
        wf['links'].append([lid_c, new_id, 0, 62, bg_input_slot, 'IMAGE'])
        embeds_62['inputs'][bg_input_slot]['link'] = lid_c
        new_node['outputs'][0]['links'].append(lid_c)

    return new_id


def find_node_by_id(wf, nid):
    for n in wf['nodes']:
        if n['id'] == nid:
            return n
    return None


def set_widget(node, idx, value):
    wv = node.setdefault('widgets_values', [])
    while len(wv) <= idx:
        wv.append(None)
    wv[idx] = value


def rewire_sampler_from_embeds62(wf):
    """WanVideoSampler брал image_embeds из Any Switch #276 (между #62 и #274).
    Any Switch удалён → надо перекоммутировать WanVideoSampler.image_embeds ← #62."""
    sampler = find_node_by_id(wf, 27)
    embeds62 = find_node_by_id(wf, 62)
    if not sampler or not embeds62:
        return

    # Найти output slot WanVideoAnimateEmbeds (image_embeds)
    embeds_out_slot = 0  # обычно первый
    # Найти input slot WanVideoSampler.image_embeds
    sampler_in_slot = None
    for i, inp in enumerate(sampler.get('inputs', [])):
        if inp.get('name') == 'image_embeds':
            sampler_in_slot = i
            break
    if sampler_in_slot is None:
        return

    # Убрать старый link у этого input (если был)
    old_lid = sampler['inputs'][sampler_in_slot].get('link')
    if old_lid:
        wf['links'] = [L for L in wf['links'] if L[0] != old_lid]

    # Создать новый link
    new_lid = max((L[0] for L in wf['links']), default=0) + 1
    wf['links'].append([new_lid, 62, embeds_out_slot, 27, sampler_in_slot, 'WANVIDIMAGE_EMBEDS'])
    sampler['inputs'][sampler_in_slot]['link'] = new_lid

    # Также обеспечить что output #62 знает про этот link
    embeds_outputs = embeds62.setdefault('outputs', [])
    if embeds_outputs:
        embeds_outputs[0].setdefault('links', []).append(new_lid)


def apply_params(wf):
    """Параметры под 9сек @ 30fps 540×960, face_strength 0.8."""

    # INTConstant Width (#150) = 540
    n = find_node_by_id(wf, 150)
    if n:
        set_widget(n, 0, TARGET_WIDTH)

    # INTConstant Height (#151) = 960
    n = find_node_by_id(wf, 151)
    if n:
        set_widget(n, 0, TARGET_HEIGHT)

    # WanVideoAnimateEmbeds #62
    # widgets: [width, height, num_frames, force_offload, frame_window_size, colormatch,
    #           pose_strength, face_strength, tiled_vae]
    n = find_node_by_id(wf, 62)
    if n:
        set_widget(n, 0, TARGET_WIDTH)
        set_widget(n, 1, TARGET_HEIGHT)
        set_widget(n, 2, TARGET_FRAMES)
        set_widget(n, 3, False)                  # force_offload
        set_widget(n, 4, FRAME_WINDOW)
        set_widget(n, 5, 'disabled')
        set_widget(n, 6, POSE_STRENGTH)
        set_widget(n, 7, FACE_STRENGTH)
        set_widget(n, 8, False)

    # ImageResizeKJv2 #64 — resize reference photo до 540×960
    n = find_node_by_id(wf, 64)
    if n:
        wv = n.get('widgets_values', [])
        # [width, height, upscale_method, keep_proportion, divisible_by, pad_color, crop, ...]
        if len(wv) >= 2:
            wv[0] = TARGET_WIDTH
            wv[1] = TARGET_HEIGHT

    # VHS_LoadVideo #63 — не трогаем widgets (пользователь подставит файл)

    # Финальный VHS_VideoCombine #214
    n = find_node_by_id(wf, 214)
    if n:
        # widgets: [frame_rate, loop_count, filename_prefix, format, ...]
        wv = n.get('widgets_values', {})
        if isinstance(wv, dict):
            wv['frame_rate'] = TARGET_FPS
            wv['filename_prefix'] = 'car_dance'
            wv['format'] = 'video/h264-mp4'
        elif isinstance(wv, list) and wv:
            wv[0] = TARGET_FPS

    # GrowMaskWithBlur #182 — подстройка мягкости краёв
    # widgets: [expand, incremental, tapered_corners, flip_input, blur_radius, lerp_alpha, decay_factor, fill_holes]
    n = find_node_by_id(wf, 182)
    if n:
        wv = n.get('widgets_values', [])
        if len(wv) >= 5:
            wv[0] = 6              # expand mask 6 px (волосы, капюшон)
            wv[4] = 10.0           # blur 10 px — мягкие края, меньше halo

    # WanVideoLoraSelect #295 — Animate Relight Lora (оставляем дефолтом или как было)
    # WanVideoLoraSelectMulti #301 — тут 4 LoRA для motion, оставляем дефолтом OFMTech

    # WanVideoSampler #27 — steps
    n = find_node_by_id(wf, 27)
    if n:
        wv = n.get('widgets_values', [])
        # widgets: [steps, cfg, shift, seed, force_offload, scheduler, riflex_freq_index, denoise_strength, batched_cfg, rope_function]
        if len(wv) >= 1:
            wv[0] = 6              # 6 steps (Lightx2v 4-step + небольшой запас)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wf = load_workflow(SRC)
    print(f"Loaded {SRC.name}: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    # 1. Резолвим Set/GetNode → прямые ссылки
    resolve_set_get(wf)
    print(f"After Set/GetNode resolve: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    # 2. Удаляем ненужные ноды (Camera уже нет в этом workflow; multi-shot, RIFE, previews)
    prune_nodes(wf, NODES_TO_REMOVE)
    print(f"After pruning: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    # 3. Чистим orphan link refs
    clean_orphan_input_links(wf)

    # 4. Перекоммутируем WanVideoSampler.image_embeds ← #62 (раньше через Any Switch)
    rewire_sampler_from_embeds62(wf)

    # 5. Добавить репликацию reference photo → bg_images (static background через N кадров)
    new_id = add_image_batch_repeat(wf)
    print(f"After rewire + bg-repeat (added #{new_id}): "
          f"{len(wf['nodes'])} nodes, {len(wf['links'])} links")

    # 6. Применяем параметры
    apply_params(wf)

    # 6. Сохраняем
    json.dump(wf, open(OUT_UI, 'w'), indent=2, ensure_ascii=False)
    print(f"Saved UI: {OUT_UI}")

    # Итоговая карта типов нод
    from collections import Counter
    types = Counter(n['type'] for n in wf['nodes'])
    print("\nFinal node type counts:")
    for t, c in types.most_common():
        print(f"  {c:3} × {t}")


if __name__ == '__main__':
    main()
