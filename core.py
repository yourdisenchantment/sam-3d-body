import os
import sys
import gc
import torch
import numpy as np
import trimesh
import trimesh.creation
import trimesh.util
from pathlib import Path
from loguru import logger

sys.path.append(os.getcwd())
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

try:
    import config_skeleton as cfg
except ImportError:
    logger.error("Запустите generate_config.py!")
    raise

try:
    from tools.build_detector import HumanDetector

    HAS_DETECTOR = True
except ImportError:
    HAS_DETECTOR = False


class BodyReconstructor:
    def __init__(self, checkpoint_dir="checkpoints/sam-3d-body-dinov3", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self._load_model()

    def _load_model(self):
        logger.info(f"⏳ Загрузка модели на {self.device}...")
        files = sorted(
            list(self.checkpoint_dir.glob("*.ckpt"))
            + list(self.checkpoint_dir.glob("*.pth")),
            key=lambda x: x.stat().st_size,
            reverse=True,
        )

        mhr_path = self.checkpoint_dir / "assets" / "mhr_model.pt"
        if not mhr_path.exists():
            mhr_path = self.checkpoint_dir / "mhr_model.pt"

        model, cfg_model = load_sam_3d_body(
            str(files[0]), device=self.device, mhr_path=str(mhr_path)
        )
        det = HumanDetector(name="vitdet", device=self.device) if HAS_DETECTOR else None
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model, model_cfg=cfg_model, human_detector=det
        )
        logger.success("🚀 Система готова")

    def _create_points_mesh(self, joints, indices, color, radius):
        """Создает облако сфер для заданных индексов"""
        parts = []
        for idx in indices:
            if idx < len(joints):
                sphere = trimesh.creation.icosphere(radius=radius, subdivisions=1)
                sphere.apply_translation(joints[idx])
                sphere.visual.face_colors = color
                parts.append(sphere)
        return trimesh.util.concatenate(parts) if parts else None

    def process(self, image_path, output_dir=None):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = {
            "json_data": [],
            "raw_meshes": {"body": [], "mhr": [], "sap_body": [], "sap_face": []},
        }

        try:
            with torch.inference_mode():
                outputs = self.estimator.process_one_image(
                    str(image_path), bbox_thr=0.5, inference_type="body"
                )

            if not outputs:
                return result

            rot_matrix = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0]
            )
            R_3x3 = rot_matrix[:3, :3]

            for i, person in enumerate(outputs):
                cam_t = person.get("pred_cam_t", np.zeros(3))
                if isinstance(cam_t, torch.Tensor):
                    cam_t = cam_t.detach().cpu().numpy().flatten()

                # Данные для JSON
                person_json = {"id": i, "mhr_joints": {}, "sapiens_joints": {}}

                # --- 1. MHR (ВНУТРЕННИЙ) ---
                if "pred_joint_coords" in person:
                    pts = person["pred_joint_coords"]
                    if isinstance(pts, torch.Tensor):
                        pts = pts.detach().cpu().numpy()[0]
                    pts_world = (pts + cam_t) @ R_3x3.T

                    # Заполняем JSON именами
                    for idx in cfg.MHR_INDICES:
                        if idx < len(pts_world):
                            name = cfg.MHR_NAMES.get(idx, f"MHR_{idx}")
                            person_json["mhr_joints"][name] = pts_world[idx].tolist()

                    # Создаем меш точек
                    mesh = self._create_points_mesh(
                        pts_world,
                        cfg.MHR_INDICES,
                        cfg.STYLE["mhr_color"],
                        cfg.STYLE["mhr_radius"],
                    )
                    if mesh:
                        result["raw_meshes"]["mhr"].append(mesh)

                # --- 2. SAPIENS (ПОВЕРХНОСТЬ) ---
                if "pred_keypoints_3d" in person:
                    pts = person["pred_keypoints_3d"]
                    if isinstance(pts, torch.Tensor):
                        pts = pts.detach().cpu().numpy()[0]
                    pts_world = (pts + cam_t) @ R_3x3.T

                    # JSON
                    for idx in range(len(pts_world)):
                        name = cfg.SAPIENS_NAMES.get(idx, f"SAP_{idx}")
                        person_json["sapiens_joints"][name] = pts_world[idx].tolist()

                    # Меш тела (0-64)
                    m_body = self._create_points_mesh(
                        pts_world,
                        cfg.SAPIENS_BODY_INDICES,
                        cfg.STYLE["sap_body_color"],
                        cfg.STYLE["sap_body_radius"],
                    )
                    if m_body:
                        result["raw_meshes"]["sap_body"].append(m_body)

                    # Меш лица (65-307)
                    m_face = self._create_points_mesh(
                        pts_world,
                        cfg.SAPIENS_FACE_INDICES,
                        cfg.STYLE["sap_face_color"],
                        cfg.STYLE["sap_face_radius"],
                    )
                    if m_face:
                        result["raw_meshes"]["sap_face"].append(m_face)

                # --- 3. МЭШ КОЖИ ---
                v = person.get("pred_vertices")
                f = self.estimator.faces
                if v is not None:
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().numpy()[0]
                    if isinstance(f, torch.Tensor):
                        f = f.detach().cpu().numpy()
                    v_final = (v + cam_t) @ R_3x3.T
                    body = trimesh.Trimesh(vertices=v_final, faces=f)
                    body.visual.face_colors = [
                        200,
                        200,
                        200,
                        100,
                    ]  # Полупрозрачный серый
                    result["raw_meshes"]["body"].append(body)

                result["json_data"].append(person_json)

        except Exception as e:
            logger.error(f"Error: {e}")

        return result
