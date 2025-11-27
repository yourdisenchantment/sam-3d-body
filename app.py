import sys
import os
import shutil
import time
import gc
import gradio as gr
import torch
import cv2
import json
import numpy as np
import trimesh
import trimesh.creation
import trimesh.util
import uuid
from pathlib import Path
from loguru import logger  # Красивые логи

# Настройка логера
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
)

sys.path.append(os.getcwd())
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

# Детектор
try:
    from tools.build_detector import HumanDetector

    HAS_DETECTOR = True
except ImportError:
    HAS_DETECTOR = False

# --- КОНФИГУРАЦИЯ ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("checkpoints/sam-3d-body-dinov3")
OUTPUT_ROOT = Path("output")
OUTPUT_ROOT.mkdir(exist_ok=True)

# Топология
SMPL_PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
]
KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    9: "left_hip",
    10: "right_hip",
    69: "neck",
    7: "left_elbow",
    8: "right_elbow",
    62: "left_wrist",
    41: "right_wrist",
    11: "left_knee",
    12: "right_knee",
    13: "left_ankle",
    14: "right_ankle",
    15: "left_big_toe",
    16: "left_small_toe",
    17: "left_heel",
    18: "right_big_toe",
    19: "right_small_toe",
    20: "right_heel",
}
NAME_TO_IDX = {v: k for k, v in KEYPOINT_NAMES.items()}
LINKS_NAMES = [
    ("left_hip", "right_hip"),
    ("left_hip", "left_shoulder"),
    ("right_hip", "right_shoulder"),
    ("left_shoulder", "neck"),
    ("right_shoulder", "neck"),
    ("neck", "nose"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_big_toe"),
    ("left_big_toe", "left_small_toe"),
    ("left_heel", "left_small_toe"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_big_toe"),
    ("right_big_toe", "right_small_toe"),
    ("right_heel", "right_small_toe"),
]
SKELETON_EDGES = [
    (NAME_TO_IDX[s], NAME_TO_IDX[e])
    for s, e in LINKS_NAMES
    if s in NAME_TO_IDX and e in NAME_TO_IDX
]

COLOR_SKIN = [200, 200, 255, 120]
COLOR_SKELETON = [255, 50, 50, 255]


# --- MONITORING UTILS ---
def log_gpu_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU VRAM: Used {allocated:.2f} GB | Reserved {reserved:.2f} GB")


def free_memory():
    """Принудительная очистка памяти"""
    gc.collect()
    torch.cuda.empty_cache()


# --- ЗАГРУЗКА ---
def find_paths():
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"Нет папки {CHECKPOINT_DIR}")
    files = sorted(
        list(CHECKPOINT_DIR.glob("*.ckpt"))
        + list(CHECKPOINT_DIR.glob("*.pth"))
        + list(CHECKPOINT_DIR.glob("*.safetensors")),
        key=lambda x: x.stat().st_size,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("Нет весов модели!")
    mhr = CHECKPOINT_DIR / "assets" / "mhr_model.pt"
    if not mhr.exists():
        mhr = CHECKPOINT_DIR / "mhr_model.pt"
    return str(files[0]), str(mhr)


logger.info(f"⏳ Инициализация на {DEVICE}...")
try:
    c_path, m_path = find_paths()
    model, cfg = load_sam_3d_body(c_path, device=DEVICE, mhr_path=m_path)

    det = None
    if HAS_DETECTOR:
        try:
            det = HumanDetector(name="vitdet", device=DEVICE)
            logger.success("Детектор (ViTDet) готов")
        except:
            pass

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model, model_cfg=cfg, human_detector=det
    )
    logger.success("Система готова к работе!")
    log_gpu_stats()
except Exception as e:
    logger.critical(f"Ошибка запуска: {e}")
    exit(1)


def serialize(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def create_skeleton_mesh(joints):
    parts = []
    limit = min(len(joints), 70)  # MHR70 limit
    for i in range(limit):
        if i not in KEYPOINT_NAMES:
            continue
        radius = 0.035 if i in [0, 69] else 0.025
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=1)
        sphere.apply_translation(joints[i])
        parts.append(sphere)
    for start, end in SKELETON_EDGES:
        if start < len(joints) and end < len(joints):
            bone = trimesh.creation.cylinder(
                radius=0.015, segment=[joints[start], joints[end]]
            )
            parts.append(bone)
    if not parts:
        return None
    skel = trimesh.util.concatenate(parts)
    skel.visual.face_colors = COLOR_SKELETON
    return skel


def process_single_image(image_path, is_warmup=False):
    start_time = time.time()

    if is_warmup:
        uid = "warmup"
        save_dir = OUTPUT_ROOT / "warmup"
    else:
        uid = uuid.uuid4().hex[:6]
        filename_stem = Path(image_path).stem
        save_dir = OUTPUT_ROOT / f"{filename_stem}_{uid}"

    if not is_warmup:
        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, save_dir / Path(image_path).name)
        logger.info(f"📸 Обработка: {Path(image_path).name} (ID: {uid})")

    # === INFERENCE ===
    outputs = []
    try:
        # Используем inference_mode для скорости и экономии памяти
        with torch.inference_mode():
            outputs = estimator.process_one_image(image_path, bbox_thr=0.5)
    except Exception as e:
        if not is_warmup:
            logger.error(f"Ошибка инференса: {e}")
        return None, None, None, None

    if not outputs:
        if not is_warmup:
            logger.warning("⚠️ Людей не найдено")
        return None, None, None, None

    if not is_warmup:
        logger.info(f"✅ Найдено людей: {len(outputs)}")

    # === 3D BUILD ===
    all_json = []
    meshes_full, meshes_body, meshes_skel = [], [], []

    for i, person in enumerate(outputs):
        p_json = {"id": i}
        joints_np = None
        cam_t = np.array([0, 0, 0])

        # Joints
        for key in ["pred_keypoints_3d", "pred_joints"]:
            if key in person:
                data = person[key]
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
                if len(data.shape) == 3:
                    data = data[0]
                joints_np = data
                p_json["joints_3d"] = serialize(joints_np)
                break

        # Cam
        if "pred_cam_t" in person:
            t = person["pred_cam_t"]
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().numpy()
            if len(t.shape) == 2:
                t = t[0]
            cam_t = t

        # Meshes
        v = person.get("pred_vertices")
        f = estimator.faces
        if v is not None:
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if len(v.shape) == 3:
                v = v[0]
            if isinstance(f, torch.Tensor):
                f = f.detach().cpu().numpy()

            # Сдвиг в мировые координаты
            v_world = v + cam_t
            body = trimesh.Trimesh(vertices=v_world, faces=f)
            body.visual.face_colors = COLOR_SKIN

            meshes_full.append(body)
            meshes_body.append(body)

            if joints_np is not None:
                skel = create_skeleton_mesh(joints_np + cam_t)
                if skel:
                    meshes_full.append(skel)
                    meshes_skel.append(skel)

        all_json.append(p_json)

    if is_warmup:
        return None

    # === EXPORT ===
    path_full = save_dir / "scene_full.glb"
    path_body = save_dir / "scene_body.glb"
    path_skel = save_dir / "scene_skel.glb"
    path_json = save_dir / "data.json"

    with open(path_json, "w") as f:
        json.dump(all_json, f, indent=2)

    def export_scene(m_list, path):
        if not m_list:
            return None
        scene = trimesh.Scene(m_list)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        scene.apply_transform(rot)
        scene.export(path)
        return str(path.absolute())

    p1 = export_scene(meshes_full, path_full)
    p2 = export_scene(meshes_body, path_body)
    p3 = export_scene(meshes_skel, path_skel)

    elapsed = time.time() - start_time
    logger.info(f"⏱️ Время обработки: {elapsed:.2f} сек")
    log_gpu_stats()

    # Очищаем память после тяжелого прохода (полезно для батчей)
    free_memory()

    return (
        {"full": p1, "body": p2, "skel": p3},
        str(path_json.absolute()),
        p1,
        str(path_json.absolute()),
    )


# --- GRADIO HANDLERS ---
def on_generate(img):
    if img is None:
        return None, None, None
    paths, json_path, init_glb, init_json = process_single_image(img)
    return paths, init_json, init_glb


def update_view(mode, paths_dict):
    if not paths_dict:
        return None
    map_mode = {"Full View": "full", "Body Only": "body", "Skeleton Only": "skel"}
    key = map_mode.get(mode, "full")
    return paths_dict.get(key)


# --- WARMUP ---
def warmup():
    logger.info("🔥 Начинаем прогрев (Warmup)...")
    dummy = np.zeros((1024, 1024, 3), dtype=np.uint8)
    cv2.imwrite("warmup.jpg", dummy)
    try:
        process_single_image("warmup.jpg", is_warmup=True)
        process_single_image("warmup.jpg", is_warmup=True)
        logger.success("Прогрев завершен!")
    except Exception as e:
        logger.warning(f"Ошибка прогрева: {e}")
    if os.path.exists("warmup.jpg"):
        os.remove("warmup.jpg")
    free_memory()


warmup()

# --- UI ---
with gr.Blocks(title="SAM 3D Body Pro") as demo:
    gr.Markdown("# 🧍 SAM 3D Body Workspace (Pro)")

    paths_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="filepath", label="Input")
            btn_gen = gr.Button("🚀 Generate 3D", variant="primary")
            # Кнопка отмены (работает для прерывания загрузки/очереди)
            btn_cancel = gr.Button("🛑 Stop / Clear", variant="stop")

        with gr.Column(scale=2):
            out_3d = gr.Model3D(
                label="3D Result", clear_color=[0.9, 0.9, 0.9, 1.0], interactive=True
            )
            with gr.Row():
                btn_full = gr.Button("Full View")
                btn_body = gr.Button("Body Only")
                btn_skel = gr.Button("Skeleton Only")
            out_json = gr.File(label="JSON Data")

    # События
    gen_event = btn_gen.click(
        on_generate, inputs=inp, outputs=[paths_state, out_json, out_3d]
    )

    # Кнопка Stop прерывает генерацию и очищает выходы
    btn_cancel.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])
    btn_cancel.click(
        lambda: (None, None, None, None), outputs=[inp, out_3d, out_json, paths_state]
    )

    # Переключатели
    btn_full.click(
        lambda s: update_view("Full View", s), inputs=paths_state, outputs=out_3d
    )
    btn_body.click(
        lambda s: update_view("Body Only", s), inputs=paths_state, outputs=out_3d
    )
    btn_skel.click(
        lambda s: update_view("Skeleton Only", s), inputs=paths_state, outputs=out_3d
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
