import sys
import os
import argparse
import time
import torch
import cv2
import json
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# Добавляем путь для импорта модулей проекта
sys.path.append(os.getcwd())
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

# Пытаемся импортировать детектор
try:
    from tools.build_detector import HumanDetector

    HAS_DETECTOR = True
except ImportError:
    HAS_DETECTOR = False

# --- КОНФИГУРАЦИЯ ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("checkpoints/sam-3d-body-dinov3")


def setup_logger(log_file=True):
    logger.remove()
    # Лог в консоль (краткий)
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO",
    )
    # Лог в файл (подробный)
    if log_file:
        log_path = f"batch_log_{int(time.time())}.log"
        logger.add(log_path, rotation="10 MB", level="DEBUG")
        return log_path
    return None


def find_paths():
    if not CHECKPOINT_DIR.exists():
        logger.error(f"Папка {CHECKPOINT_DIR} не найдена")
        sys.exit(1)

    files = sorted(
        list(CHECKPOINT_DIR.glob("*.ckpt"))
        + list(CHECKPOINT_DIR.glob("*.pth"))
        + list(CHECKPOINT_DIR.glob("*.safetensors")),
        key=lambda x: x.stat().st_size,
        reverse=True,
    )
    if not files:
        logger.error("Нет весов модели!")
        sys.exit(1)

    mhr = CHECKPOINT_DIR / "assets" / "mhr_model.pt"
    if not mhr.exists():
        mhr = CHECKPOINT_DIR / "mhr_model.pt"

    return str(files[0]), str(mhr)


def serialize(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def get_gpu_memory():
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB"
    return "N/A"


def filter_images(input_dir, cams):
    """
    Сканирует папку и фильтрует по камерам.
    Формат: {cam_id}_*_{frame_id}.jpeg
    """
    valid_extensions = {".jpg", ".jpeg", ".png"}
    all_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions]
    )

    if not cams:
        return all_files

    filtered_files = []
    # Нормализуем ввод камер (чтобы '1' стало '01', если файлы так называются, или ищем точное совпадение)
    # Но лучше искать по префиксу до первого '_'
    target_cams = set(cams)  # ['01', '02']

    for f in all_files:
        try:
            # Парсим имя файла: 01_016BDOG#2_00000464.jpeg -> cam_id = "01"
            cam_id = f.name.split("_")[0]
            if cam_id in target_cams:
                filtered_files.append(f)
        except Exception:
            continue  # Пропускаем файлы с неправильным неймингом

    # Проверка на ошибку (если указали камеру, а файлов нет)
    if not filtered_files:
        logger.error(f"Не найдено изображений для камер: {target_cams}")
        sys.exit(1)

    return filtered_files


def process_batch(args):
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Входная папка не найдена: {input_path}")
        sys.exit(1)

    # 1. Настройка папок вывода
    if args.output_skeletons:
        skel_dir = Path(args.output_skeletons)
    else:
        skel_dir = input_path.parent / f"{input_path.name}_skeletons"

    if args.output_meshes:
        mesh_dir = Path(args.output_meshes)
    else:
        mesh_dir = input_path.parent / f"{input_path.name}_meshes"

    skel_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    log_file = setup_logger()
    logger.info(f"📂 Вход: {input_path}")
    logger.info(f"💀 Скелеты: {skel_dir}")
    logger.info(f"🧊 Меши: {mesh_dir}")
    if log_file:
        logger.info(f"📝 Лог: {log_file}")

    # 2. Фильтрация файлов
    files_to_process = filter_images(input_path, args.cams)
    logger.info(f"📸 Найдено изображений для обработки: {len(files_to_process)}")

    # 3. Загрузка модели
    logger.info("⏳ Загрузка модели...")
    try:
        ckpt, mhr = find_paths()
        model, cfg = load_sam_3d_body(ckpt, device=DEVICE, mhr_path=mhr)
        det = HumanDetector(name="vitdet", device=DEVICE) if HAS_DETECTOR else None
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model, model_cfg=cfg, human_detector=det
        )
        logger.success("Модель загружена!")
    except Exception as e:
        logger.critical(f"Критическая ошибка загрузки: {e}")
        sys.exit(1)

    # 4. Прогрев (Warmup) - чтобы tqdm показывал реальное время сразу
    logger.info("🔥 Прогрев GPU...")
    try:
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.imwrite("warmup_batch.jpg", dummy)
        with torch.inference_mode():
            estimator.process_one_image("warmup_batch.jpg", bbox_thr=0.5)
        os.remove("warmup_batch.jpg")
    except:
        pass

    # 5. Основной цикл
    pbar = tqdm(files_to_process, unit="img")
    success_count = 0
    skipped_count = 0
    error_count = 0

    for img_file in pbar:
        # Имена выходных файлов
        json_out = skel_dir / f"{img_file.stem}.json"
        glb_out = mesh_dir / f"{img_file.stem}.glb"

        # Пропуск готовых
        if args.skip_existing:
            if json_out.exists() and glb_out.exists():
                skipped_count += 1
                pbar.set_description(f"Skip {img_file.name}")
                continue

        pbar.set_description(f"Proc {img_file.name}")

        try:
            # Inference
            with torch.inference_mode():
                outputs = estimator.process_one_image(str(img_file), bbox_thr=0.5)

            if not outputs:
                logger.warning(f"На фото {img_file.name} люди не найдены.")
                # Можно создать пустой JSON, чтобы отметить факт обработки
                with open(json_out, "w") as f:
                    json.dump([], f)
                continue

            # Собираем данные
            all_people_data = []
            scene_meshes = []

            for i, person in enumerate(outputs):
                p_data = {"id": i}
                joints_np = None
                cam_t = np.array([0, 0, 0])

                # Joints extraction
                for key in ["pred_keypoints_3d", "pred_joints"]:
                    if key in person:
                        data = person[key]
                        if isinstance(data, torch.Tensor):
                            data = data.detach().cpu().numpy()
                        if len(data.shape) == 3:
                            data = data[0]
                        joints_np = data
                        p_data["joints_3d"] = serialize(joints_np)
                        break

                # Cam translation
                if "pred_cam_t" in person:
                    t = person["pred_cam_t"]
                    if isinstance(t, torch.Tensor):
                        t = t.detach().cpu().numpy()
                    if len(t.shape) == 2:
                        t = t[0]
                    cam_t = t

                # Mesh extraction
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
                    # Цвет кожи для наглядности (если открыть в просмотрщике)
                    body.visual.face_colors = [200, 200, 255, 255]
                    scene_meshes.append(body)

                all_people_data.append(p_data)

            # Сохранение JSON (Скелеты)
            with open(json_out, "w") as f:
                json.dump(all_people_data, f, indent=2)

            # Сохранение GLB (Меши)
            if scene_meshes:
                scene = trimesh.Scene(scene_meshes)
                # Поворот для совместимости с GLB Y-up
                rot = trimesh.transformations.rotation_matrix(
                    np.radians(180), [1, 0, 0]
                )
                scene.apply_transform(rot)
                scene.export(glb_out)

            success_count += 1

            # Обновление статистики в баре
            if success_count % 10 == 0:
                pbar.set_postfix({"VRAM": get_gpu_memory(), "Done": success_count})

        except Exception as e:
            error_count += 1
            logger.error(f"Ошибка при обработке {img_file.name}: {e}")
            continue

    logger.info("🏁 Обработка завершена!")
    logger.info(f"✅ Успешно: {success_count}")
    logger.info(f"⏭️ Пропущено: {skipped_count}")
    logger.info(f"❌ Ошибки: {error_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Фоновая обработка SAM 3D Body")

    # Обязательный аргумент
    parser.add_argument("--input", type=str, required=True, help="Путь к папке с фото")

    # Опциональные пути
    parser.add_argument("--output-skeletons", type=str, help="Папка для JSON скелетов")
    parser.add_argument("--output-meshes", type=str, help="Папка для GLB мешей")

    # Фильтры
    parser.add_argument(
        "--cams",
        nargs="+",
        type=str,
        help="Список номеров камер (напр. '01' '02'). Если не задано - все.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Пропускать уже обработанные файлы"
    )

    args = parser.parse_args()

    process_batch(args)
