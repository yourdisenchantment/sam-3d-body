import sys
from pathlib import Path
from sam_3d_body import load_sam_3d_body


def check_system():
    print("\n🧪 [Check] Начинаем проверку системы...")

    # 1. Проверка путей
    base_dir = Path.cwd()
    checkpoint_dir = base_dir / "checkpoints" / "sam-3d-body-dinov3"

    print(f"   📂 Папка чекпоинтов: {checkpoint_dir}")

    if not checkpoint_dir.exists():
        print("   ❌ Ошибка: Папка не найдена. Веса не скачались?")
        sys.exit(1)

    # Ищем файлы весов
    extensions = ["*.ckpt", "*.pth", "*.safetensors"]
    files = []
    for ext in extensions:
        files.extend(list(checkpoint_dir.glob(ext)))

    if not files:
        print("   ❌ Ошибка: Файлы весов (ckpt/pth) отсутствуют.")
        sys.exit(1)

    # Сортируем по размеру
    files.sort(key=lambda x: x.stat().st_size, reverse=True)
    ckpt_path = files[0]
    print(
        f"   ✅ Найдены веса: {ckpt_path.name} ({ckpt_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB)"
    )

    # Ищем Asset (MHR)
    mhr_path = checkpoint_dir / "assets" / "mhr_model.pt"
    if not mhr_path.exists():
        # Проверка альтернативного пути
        mhr_path = checkpoint_dir / "mhr_model.pt"

    if not mhr_path.exists():
        print("   ❌ Ошибка: Не найден вспомогательный файл mhr_model.pt")
        sys.exit(1)

    print(f"   ✅ Найден Asset: {mhr_path.name}")

    # 2. Попытка загрузки
    print("   ⏳ Попытка загрузки модели в память...")
    try:
        # Используем CPU для теста совместимости
        device = "cpu"
        print(f"      Device: {device}")

        model, cfg = load_sam_3d_body(
            str(ckpt_path), device=device, mhr_path=str(mhr_path)
        )
        print("   ✅ [Check] УСПЕХ! Модель инициализирована корректно.\n")
    except Exception as e:
        print(f"\n   ❌ [Check] КРИТИЧЕСКАЯ ОШИБКА ЗАГРУЗКИ: {e}")
        sys.exit(1)


if __name__ == "__main__":
    check_system()
