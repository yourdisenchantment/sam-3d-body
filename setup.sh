#!/bin/bash
set -e  # Остановить выполнение при любой ошибке

# --- НАСТРОЙКИ ДЛЯ КОМПИЛЯЦИИ ---
# Готовим Detectron2 под архитектуру Ampere (RTX 3090),
# даже если сейчас стоит другая карта.
export TORCH_CUDA_ARCH_LIST="8.6"
export FORCE_CUDA="1"

print_header() {
    echo ""
    echo "========================================================"
    echo "   $1"
    echo "========================================================"
    echo ""
}

print_header "🚀 СТАРТ УСТАНОВКИ SAM 3D BODY"

# 1. Очистка
echo "🧹 Шаг 1: Очистка старого окружения..."
rm -rf .venv uv.lock checkpoints

# 2. Venv
echo "🐍 Шаг 2: Создание venv (Python 3.11)..."
uv venv --python 3.11
source .venv/bin/activate

# 3. PyTorch
print_header "🔥 Шаг 3: Установка PyTorch (CUDA 12.4)"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Основные библиотеки
print_header "📚 Шаг 4: Установка зависимостей"
uv pip install pytorch-lightning pyrender opencv-python yacs scikit-image \
    einops timm dill pandas rich hydra-core hydra-submitit-launcher \
    hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma \
    joblib seaborn wandb appdirs ffmpeg cython jsonlines pytest \
    xtcocotools loguru optree fvcore black pycocotools tensorboard \
    huggingface_hub hf_transfer jupyter gradio trimesh matplotlib scipy

# 5. Detectron2 & MoGe
print_header "⚙️ Шаг 5: Сборка Detectron2 и MoGe"
uv pip install "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9" --no-build-isolation
uv pip install git+https://github.com/microsoft/MoGe.git

# 6. Установка проекта
print_header "🔗 Шаг 6: Регистрация sam-3d-body"
# pyproject.toml должен уже лежать в папке (мы его создали вручную)
uv pip install -e .

# 7. Фикс DINOv3
print_header "🔧 Шаг 7: Исправление кэша DINOv3"
uv run python fix_dinov3_cache.py

# 8. Скачивание весов
print_header "⬇️ Шаг 8: Скачивание весов с Hugging Face"
echo "🔑 Пожалуйста, войдите в аккаунт HF (вставьте токен):"
uv run python -c "from huggingface_hub import login; login()"

echo "📡 Скачивание файлов (Turbo mode)..."
mkdir -p checkpoints
HF_HUB_ENABLE_HF_TRANSFER=1 uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/sam-3d-body-dinov3', local_dir='checkpoints/sam-3d-body-dinov3')"

# 9. Проверка
print_header "🧪 Шаг 9: Финальная проверка"
uv run python check_installation.py

print_header "🎉 УСТАНОВКА УСПЕШНО ЗАВЕРШЕНА!"
echo "Теперь вы можете запускать: uv run python app.py"
