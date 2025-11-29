import os
import sys
import time
import gc
import torch
import cv2
import numpy as np
import trimesh
import trimesh.creation
import trimesh.util
import psutil
import subprocess
from pathlib import Path
from loguru import logger

sys.path.append(os.getcwd())
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

try:
    from tools.build_detector import HumanDetector

    HAS_DETECTOR = True
except ImportError:
    HAS_DETECTOR = False

# --- ТОПОЛОГИЯ ---
KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_hip",
    10: "right_hip",
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
    41: "right_wrist",
    62: "left_wrist",
    69: "neck",
}
NAME_TO_IDX = {v: k for k, v in KEYPOINT_NAMES.items()}

LINKS_NAMES = [
    # ТОРС
    ("left_hip", "right_hip"),
    ("left_hip", "left_shoulder"),
    ("right_hip", "right_shoulder"),
    ("left_shoulder", "neck"),
    ("right_shoulder", "neck"),
    # ГОЛОВА (Добавлена трапеция для красоты)
    ("neck", "nose"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    # ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"), # Опционально: трапеции
    # РУКИ
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # НОГИ
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    # СТОПЫ (ИСПРАВЛЕНО: Замыкаем в треугольник)
    # Левая
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_big_toe"),
    ("left_ankle", "left_small_toe"),  # <--- Добавлено: лодыжка к мизинцу
    # ("left_big_toe", "left_small_toe"),  # <--- Добавлено: связка пальцев
    # ("left_heel", "left_small_toe"),  # <--- Добавлено: пятка к мизинцу
    # Правая
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_big_toe"),
    ("right_ankle", "right_small_toe"),  # <--- Добавлено
    # ("right_big_toe", "right_small_toe"),  # <--- Добавлено
    # ("right_heel", "right_small_toe"),  # <--- Добавлено
]

SKELETON_EDGES = []
for start, end in LINKS_NAMES:
    if start in NAME_TO_IDX and end in NAME_TO_IDX:
        SKELETON_EDGES.append((NAME_TO_IDX[start], NAME_TO_IDX[end]))

COLOR_JOINTS = [255, 0, 0, 255]
COLOR_BONES = [180, 180, 180, 255]
COLOR_SKIN = [200, 200, 255, 120]


class BodyReconstructor:
    def __init__(self, checkpoint_dir="checkpoints/sam-3d-body-dinov3", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)

        # Логгер настраивается снаружи
        self._load_model()

    def _load_model(self):
        files = sorted(
            list(self.checkpoint_dir.glob("*.ckpt"))
            + list(self.checkpoint_dir.glob("*.pth")),
            key=lambda x: x.stat().st_size,
            reverse=True,
        )
        if not files:
            raise FileNotFoundError("Веса не найдены!")

        mhr_path = self.checkpoint_dir / "assets" / "mhr_model.pt"
        if not mhr_path.exists():
            mhr_path = self.checkpoint_dir / "mhr_model.pt"

        model, cfg = load_sam_3d_body(
            str(files[0]), device=self.device, mhr_path=str(mhr_path)
        )
        det = HumanDetector(name="vitdet", device=self.device) if HAS_DETECTOR else None

        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model, model_cfg=cfg, human_detector=det
        )

    def get_system_stats(self):
        """Сбор детальной статистики железа"""
        stats = {
            "cpu_util": f"{psutil.cpu_percent()}%",
            "ram_used": f"{psutil.virtual_memory().used / (1024**3):.1f}GB",
            # Дефолтные значения для GPU
            "gpu_temp": "N/A",
            "gpu_power": "N/A",
            "gpu_util": "N/A",
            "gpu_mem_used": "N/A",
            "gpu_mem_total": "N/A",
        }

        if torch.cuda.is_available():
            try:
                # nvidia-smi query: temp, power, util, mem_used, mem_total
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ]
                output = subprocess.check_output(cmd).decode("utf-8").strip().split(",")

                if len(output) >= 5:
                    stats["gpu_temp"] = f"{output[0].strip()}C"
                    stats["gpu_power"] = f"{float(output[1].strip()):.1f}W"
                    stats["gpu_util"] = f"{output[2].strip()}%"
                    stats["gpu_mem_used"] = f"{float(output[3].strip()) / 1024:.1f}GB"
                    stats["gpu_mem_total"] = f"{float(output[4].strip()) / 1024:.1f}GB"
            except Exception:
                # Fallback если nvidia-smi недоступен (например, в контейнере без прав)
                mem = torch.cuda.memory_allocated() / (1024**3)
                stats["gpu_mem_used"] = f"{mem:.1f}GB (Torch)"

        return stats

    def _create_skeleton_mesh(self, joints):
        parts = []
        for idx in KEYPOINT_NAMES.keys():
            if idx < len(joints):
                r = 0.03 if idx in [0, 69] else 0.025
                sphere = trimesh.creation.icosphere(radius=r, subdivisions=1)
                sphere.apply_translation(joints[idx])
                sphere.visual.face_colors = COLOR_JOINTS
                parts.append(sphere)

        for s, e in SKELETON_EDGES:
            if s < len(joints) and e < len(joints):
                bone = trimesh.creation.cylinder(
                    radius=0.01, segment=[joints[s], joints[e]]
                )
                bone.visual.face_colors = COLOR_BONES
                parts.append(bone)

        if not parts:
            return None
        return trimesh.util.concatenate(parts)

    def process(self, image_path, output_dir=None):
        start_time = time.time()
        result_data = {"json_data": [], "scene_body": None, "stats": {}}

        try:
            # Inference
            with torch.inference_mode():
                outputs = self.estimator.process_one_image(
                    str(image_path), bbox_thr=0.5, inference_type="body"
                )

            if not outputs:
                return result_data

            meshes_body = []
            rot_matrix = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0]
            )

            for i, person in enumerate(outputs):
                # Joints
                joints = None
                for key in ["pred_keypoints_3d", "pred_joints"]:
                    if key in person:
                        joints = person[key]
                        if isinstance(joints, torch.Tensor):
                            joints = joints.detach().cpu().numpy()
                        if len(joints.shape) == 3:
                            joints = joints[0]
                        break

                # Cam shift
                cam_t = np.array([0, 0, 0])
                if "pred_cam_t" in person:
                    t = person["pred_cam_t"]
                    if isinstance(t, torch.Tensor):
                        t = t.detach().cpu().numpy()
                    if len(t.shape) == 2:
                        t = t[0]
                    cam_t = t

                # Data collection
                p_data = {"id": i}
                if joints is not None:
                    p_data["joints_3d"] = joints.tolist()
                result_data["json_data"].append(p_data)

                # Mesh building
                v = person.get("pred_vertices")
                f = self.estimator.faces

                if v is not None:
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().numpy()
                    if len(v.shape) == 3:
                        v = v[0]
                    if isinstance(f, torch.Tensor):
                        f = f.detach().cpu().numpy()

                    body = trimesh.Trimesh(vertices=v + cam_t, faces=f)
                    body.visual.face_colors = COLOR_SKIN
                    meshes_body.append(body)

            # Create scene
            def make_scene(ms):
                if not ms:
                    return None
                s = trimesh.Scene(ms)
                s.apply_transform(rot_matrix)
                return s

            result_data["scene_body"] = make_scene(meshes_body)

            # Stats
            elapsed = time.time() - start_time
            sys_stats = self.get_system_stats()
            result_data["stats"] = sys_stats
            result_data["stats"]["time_sec"] = f"{elapsed:.2f}"

        except Exception as e:
            # Логгируем ошибку снаружи, здесь просто возвращаем пустой результат
            result_data["error"] = str(e)

        self._cleanup()
        return result_data

    def _cleanup(self):
        gc.collect()
