import argparse
import sys
import json
import torch
import time
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# Импорт ядра
from core import BodyReconstructor


def setup_logger(log_file):
    logger.remove()
    # В КОНСОЛЬ: Только ошибки, чтобы не мешать tqdm
    logger.add(sys.stderr, format="<red>{level}</red>: {message}", level="ERROR")

    # В ФАЙЛ: Подробно всё (INFO)
    # Формат: Время | Уровень | Сообщение
    logger.add(
        log_file,
        rotation="20 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
    )


def main():
    parser = argparse.ArgumentParser(description="Пакетная обработка SAM 3D Body")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Путь к корневой папке датасета"
    )
    parser.add_argument("--cam", type=str, help="Фильтр по номеру камеры")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    rgb_dir = dataset_dir / "rgb"
    mesh_dir = dataset_dir / "mesh"
    skel_dir = dataset_dir / "skeleton"

    # Создаем уникальный лог-файл для каждого запуска
    log_file = dataset_dir / f"process_log_{int(time.time())}.log"

    if not dataset_dir.exists() or not rgb_dir.exists():
        print(f"❌ Ошибка: папка {rgb_dir} не найдена")
        sys.exit(1)

    mesh_dir.mkdir(parents=True, exist_ok=True)
    skel_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(log_file)
    logger.info(f"=== СТАРТ ОБРАБОТКИ ===")
    logger.info(f"Датасет: {dataset_dir}")
    logger.info(f"Устройство: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Поиск файлов
    all_files = sorted(
        [f for f in rgb_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )
    if args.cam:
        files_to_process = [f for f in all_files if f.name.startswith(f"{args.cam}_")]
    else:
        files_to_process = all_files

    logger.info(f"Найдено файлов: {len(files_to_process)}")
    print(f"🚀 Найдено {len(files_to_process)} файлов. Лог пишется в: {log_file.name}")

    # Инициализация
    try:
        reconstructor = BodyReconstructor()
        logger.info("Модель загружена успешно")
    except Exception as e:
        logger.critical(f"Ошибка инициализации: {e}")
        sys.exit(1)

    success = 0
    errors = 0
    skipped = 0

    # TQDM для визуализации в терминале
    pbar = tqdm(files_to_process, unit="img", dynamic_ncols=True)

    for i, img_path in enumerate(pbar):
        stem = img_path.stem

        glb_out = mesh_dir / f"{stem}.glb"
        json_out = skel_dir / f"{stem}.json"

        # Пропуск
        if args.skip_existing and glb_out.exists() and json_out.exists():
            skipped += 1
            continue

        # Обработка
        result = reconstructor.process(img_path)

        # Получаем статистику для логов
        stats = result.get("stats", {})
        error = result.get("error", None)

        # Обновляем TQDM (только самое важное для глаз)
        # Показываем температуру GPU, чтобы следить за перегревом
        gpu_temp = stats.get("gpu_temp", "N/A")
        pbar.set_description(f"GPU: {gpu_temp}")

        if error:
            logger.error(f"Файл: {stem} | Ошибка: {error} | Stats: {stats}")
            errors += 1
            continue

        if result["scene_body"] is None:
            logger.warning(f"Файл: {stem} | Людей не найдено")
            continue

        # Сохранение
        try:
            result["scene_body"].export(glb_out)
            with open(json_out, "w") as f:
                json.dump(result["json_data"], f, indent=2)

            success += 1

            # ЗАПИСЬ В ЛОГ ФАЙЛ (Полная инфа)
            # Пример: Файл: img_01 | Time: 0.45s | GPU: 55C, 120W, 98%, 4.5/24GB | RAM: 5.2GB
            log_msg = (
                f"Файл: {stem} | "
                f"Time: {stats.get('time_sec')}s | "
                f"GPU: {stats.get('gpu_temp')}, {stats.get('gpu_power')}, {stats.get('gpu_util')}, {stats.get('gpu_mem_used')}/{stats.get('gpu_mem_total')} | "
                f"RAM: {stats.get('ram_used')}"
            )
            logger.info(log_msg)

        except Exception as e:
            logger.error(f"Файл: {stem} | Ошибка сохранения: {e}")
            errors += 1

        # Защита от перегрева и переполнения памяти (каждые 50 кадров)
        if i > 0 and i % 50 == 0:
            torch.cuda.empty_cache()
            # time.sleep(0.1) # Можно раскомментировать, если греется

    # Финал
    final_msg = (
        f"=== ЗАВЕРШЕНО === Успех: {success}, Ошибок: {errors}, Пропущено: {skipped}"
    )
    print(f"\n{final_msg}")
    logger.info(final_msg)


if __name__ == "__main__":
    main()
