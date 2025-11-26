import zipfile
import shutil
import sys
import subprocess
from pathlib import Path


def run_command(command):
    """Запускает системную команду"""
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при выполнении команды: {command}")
        sys.exit(1)


def fix_cache():
    # Пути через pathlib
    user_home = Path.home()
    target_dir = user_home / ".cache" / "torch" / "hub" / "facebookresearch_dinov3_main"
    zip_filename = "dinov3.zip"
    zip_path = Path.cwd() / zip_filename
    temp_extract_dir = Path.cwd() / "dinov3_temp_extract"
    repo_url = "https://github.com/facebookresearch/dinov3/archive/refs/heads/main.zip"

    print("\n🔧 [Fix Cache] Начинаем исправление кэша DINOv3...")
    print(f"   📂 Целевая директория: {target_dir}")

    # 1. Скачивание
    if not zip_path.exists():
        print(f"   ⬇️ Скачиваем архив с {repo_url}...")
        run_command(f"wget {repo_url} -O {zip_filename}")
    else:
        print(f"   ✅ Архив {zip_filename} уже существует.")

    # 2. Очистка старого кэша
    if target_dir.exists():
        print("   🧹 Удаляем старую версию кэша...")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    # 3. Распаковка
    print(f"   📦 Распаковываем {zip_filename}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_extract_dir)
    except zipfile.BadZipFile:
        print("   ❌ Ошибка: Архив поврежден. Удалите его и попробуйте снова.")
        sys.exit(1)

    # 4. Перемещение файлов
    source_folder = temp_extract_dir / "dinov3-main"
    if not source_folder.exists():
        print("   ❌ Ошибка: В архиве нет папки dinov3-main.")
        sys.exit(1)

    print("   🚚 Перемещаем файлы в кэш Torch...")
    for item in source_folder.iterdir():
        # shutil.move требует стринги в старых версиях, но pathlib тоже ок в новых.
        # Для надежности используем str()
        shutil.move(str(item), str(target_dir))

    # 5. Уборка
    print("   🧹 Удаляем временные файлы...")
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)

    # Опционально: удаляем zip
    # if zip_path.exists(): zip_path.unlink()

    print("   ✅ [Fix Cache] Готово! DINOv3 установлен вручную.\n")


if __name__ == "__main__":
    fix_cache()
