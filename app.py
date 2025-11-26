import sys
import os
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

# Добавляем путь для импортов
sys.path.append(os.getcwd())

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

# Пытаемся подключить детектор (так как мы ставили detectron2)
try:
    from tools.build_detector import HumanDetector

    HAS_DETECTOR = True
except ImportError:
    print("⚠️ Detectron2 не найден или HumanDetector недоступен.")
    HAS_DETECTOR = False

# --- КОНФИГУРАЦИЯ ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("checkpoints/sam-3d-body-dinov3")

# Иерархия родителей SMPL (для рисования костей)
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
    20,
    21,
]

# Цвета (R, G, B, A)
COLOR_SKELETON = [255, 50, 50, 255]  # Ярко-красный, непрозрачный
COLOR_SKIN = [200, 200, 200, 100]  # Серый, полупрозрачный


def find_paths():
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"Нет папки {CHECKPOINT_DIR}")

    # Ищем веса
    files = (
        list(CHECKPOINT_DIR.glob("*.ckpt"))
        + list(CHECKPOINT_DIR.glob("*.pth"))
        + list(CHECKPOINT_DIR.glob("*.safetensors"))
    )
    files.sort(key=lambda x: x.stat().st_size, reverse=True)
    if not files:
        raise FileNotFoundError("Нет весов модели!")

    # Ищем Asset
    mhr = CHECKPOINT_DIR / "assets" / "mhr_model.pt"
    if not mhr.exists():
        mhr = CHECKPOINT_DIR / "mhr_model.pt"

    return str(files[0]), str(mhr)


# --- ЗАГРУЗКА ---
print(f"⏳ Инициализация системы на {DEVICE}...")
try:
    c_path, m_path = find_paths()
    print(f"📂 Load: {Path(c_path).name}")

    # Загружаем модель
    model, cfg = load_sam_3d_body(c_path, device=DEVICE, mhr_path=m_path)

    # Загружаем детектор
    det = None
    if HAS_DETECTOR:
        print("🕵️ Запускаем HumanDetector (ViTDet)...")
        try:
            det = HumanDetector(name="vitdet", device=DEVICE)
            print("✅ Детектор готов!")
        except Exception as e:
            print(f"⚠️ Ошибка детектора: {e}")

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model, model_cfg=cfg, human_detector=det
    )
    print("🚀 Система полностью готова!")

except Exception as e:
    print(f"❌ Критическая ошибка запуска: {e}")
    exit(1)


# --- УТИЛИТЫ ---
def serialize(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def create_skeleton_mesh(joints):
    """Строит геометрию скелета (сферы + цилиндры)"""
    parts = []
    limit = min(len(joints), len(SMPL_PARENTS))

    for i in range(limit):
        loc = joints[i]

        # Сустав (Сфера)
        sphere = trimesh.creation.icosphere(radius=0.035, subdivisions=1)
        sphere.apply_translation(loc)
        parts.append(sphere)

        # Кость (Цилиндр)
        parent_idx = SMPL_PARENTS[i]
        if parent_idx != -1 and parent_idx < len(joints):
            bone = trimesh.creation.cylinder(
                radius=0.02, segment=[loc, joints[parent_idx]]
            )
            parts.append(bone)

    if not parts:
        return None

    # Объединяем в один меш для производительности
    skeleton = trimesh.util.concatenate(parts)
    skeleton.visual.face_colors = COLOR_SKELETON
    return skeleton


def run_inference(input_image):
    if input_image is None:
        return None, None

    # Генерируем уникальное имя, чтобы браузер не кэшировал старую модель
    uid = uuid.uuid4().hex[:6]
    temp_img = f"temp_{uid}.jpg"
    glb_out = f"result_{uid}.glb"
    json_out = f"skeleton_{uid}.json"

    print(f"\n📸 Обработка запроса {uid}...")

    # Конвертация в BGR
    cv2.imwrite(temp_img, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))

    outputs = []
    try:
        # Если есть детектор, bbox_thr отсечет мусор.
        # Если детектора нет, попытается взять всю картинку.
        outputs = estimator.process_one_image(temp_img, bbox_thr=0.5)
    except Exception as e:
        print(f"Ошибка инференса: {e}")

    # Удаляем времянку
    if os.path.exists(temp_img):
        os.remove(temp_img)

    if not outputs:
        print("⚠️ Людей не найдено.")
        return None, None

    print(f"✅ Найдено людей: {len(outputs)}")
    person = outputs[0]  # Берем первого

    # --- 1. JSON Export ---
    json_data = {"joints_3d": []}
    joints_np = None

    if "pred_joints" in person:
        json_data["joints_3d"] = serialize(person["pred_joints"])
        joints_np = person["pred_joints"].detach().cpu().numpy()
    elif "joints" in person:
        json_data["joints_3d"] = serialize(person["joints"])
        joints_np = person["joints"].detach().cpu().numpy()

    with open(json_out, "w") as f:
        json.dump(json_data, f, indent=2)

    # --- 2. 3D GLB Export ---
    v = person.get("pred_vertices")
    f = estimator.faces

    scene_meshes = []

    # Тело
    if v is not None and f is not None:
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if len(v.shape) == 3:
            v = v[0]
        if isinstance(f, torch.Tensor):
            f = f.detach().cpu().numpy()

        body = trimesh.Trimesh(vertices=v, faces=f)
        body.visual.face_colors = COLOR_SKIN  # Прозрачность
        scene_meshes.append(body)

    # Скелет
    if joints_np is not None:
        if len(joints_np.shape) == 3:
            joints_np = joints_np[0]
        skel = create_skeleton_mesh(joints_np)
        if skel:
            scene_meshes.append(skel)

    if scene_meshes:
        scene = trimesh.Scene(scene_meshes)
        # Поворачиваем, чтобы стоял вертикально (SMPL fix)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        scene.apply_transform(rot)

        scene.export(glb_out)
        print(f"💾 Модель сохранена: {glb_out}")
        return os.path.abspath(glb_out), os.path.abspath(json_out)
    else:
        return None, None


# --- UI ---
with gr.Blocks(title="SAM 3D Body") as demo:
    gr.Markdown("# 🧍 SAM 3D Body Local")
    with gr.Row():
        inp = gr.Image(type="numpy", label="Input Image")
        with gr.Column():
            # clear_color управляет фоном вьювера (светло-серый)
            out_3d = gr.Model3D(label="3D Result", clear_color=[0.9, 0.9, 0.9, 1.0])
            out_json = gr.File(label="Skeleton JSON")

    gr.Button("Generate 3D", variant="primary").click(
        run_inference, inp, [out_3d, out_json]
    )

if __name__ == "__main__":
    # share=True дает публичную ссылку
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
