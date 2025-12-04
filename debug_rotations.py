import torch
import numpy as np
import os
import sys
import roma  # Библиотека для работы с вращениями (используется в SAM3D)
from loguru import logger

# Добавляем путь к проекту
sys.path.append(os.getcwd())
from core import BodyReconstructor

# Маппинг индексов MHR (127 joints) на понятные имена
# Это нужно, чтобы понять, какой индекс за что отвечает
MHR_JOINTS_MAP = {
    0: "Pelvis (Hip)",
    1: "Spine1",
    12: "Neck",
    13: "Head",
    19: "L_Clavicle",
    20: "L_Shoulder",
    21: "L_Elbow",
    22: "L_Wrist",
    28: "R_Clavicle",
    29: "R_Shoulder",
    30: "R_Elbow",
    31: "R_Wrist",
    37: "L_Hip",
    38: "L_Knee",
    39: "L_Ankle",
    46: "R_Hip",
    47: "R_Knee",
    48: "R_Ankle",
}


def analyze_rotations(image_path):
    print(f"🚀 Запуск анализа для: {image_path}")

    # 1. Инициализация
    reconstructor = BodyReconstructor()

    # 2. Инференс (получаем словарь с результатами)
    # Используем process_one_image, который возвращает список словарей (all_out)
    outputs = reconstructor.estimator.process_one_image(
        str(image_path), bbox_thr=0.5, inference_type="body"
    )

    if not outputs:
        print("❌ Человек не обнаружен.")
        return

    # Берем первого человека
    data = outputs[0]

    # 3. Достаем глобальные вращения
    # В коде sam_3d_body_estimator.py это сохраняется как "pred_global_rots"
    # Формат: (127, 3, 3) - Numpy array
    rot_mats_np = data.get("pred_global_rots")

    if rot_mats_np is None:
        print("❌ Вращения не найдены в выходных данных.")
        return

    # Конвертируем в torch tensor для работы с roma
    rot_mats = torch.tensor(rot_mats_np)  # Shape: (127, 3, 3)

    print(f"\n📦 Raw Rotation Matrices Shape: {rot_mats.shape}")
    print("-" * 100)
    print(
        f"{'ID':<4} | {'Name':<15} | {'Quaternion (x, y, z, w)':<35} | {'Euler (XYZ) [deg]':<30}"
    )
    print("-" * 100)

    # 4. Конвертация
    # Матрицы -> Кватернионы
    quats = roma.rotmat_to_unitquat(rot_mats)  # (127, 4) -> [x, y, z, w]

    # Матрицы -> Углы Эйлера (XYZ)
    eulers_rad = roma.rotmat_to_euler("xyz", rot_mats)
    eulers_deg = eulers_rad * (180.0 / np.pi)

    # 5. Вывод данных для ключевых суставов
    for idx, name in MHR_JOINTS_MAP.items():
        if idx >= len(rot_mats):
            continue

        q = quats[idx].numpy()
        e = eulers_deg[idx].numpy()

        q_str = f"[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]"
        e_str = f"[{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}]"

        print(f"{idx:<4} | {name:<15} | {q_str:<35} | {e_str:<30}")

    print("-" * 100)
    print("\n💡 ПОЯСНЕНИЕ К CSV:")
    print("В твоем CSV есть колонки X_Hip, Y_Hip, Z_Hip.")
    print("Это соответствует колонке 'Euler (XYZ)' в таблице выше.")
    print(
        "Если тебе нужны Кватернионы (как работает движок внутри), смотри колонку 'Quaternion'."
    )

    # Пример доступа к конкретному значению для CSV
    hip_euler = eulers_deg[0].numpy()  # 0 = Pelvis/Hip
    print(f"\nCSV Format Example (Hip):")
    print(f"X_Hip: {hip_euler[0]:.7f}")
    print(f"Y_Hip: {hip_euler[1]:.7f}")
    print(f"Z_Hip: {hip_euler[2]:.7f}")


if __name__ == "__main__":
    # Укажи путь к картинке
    IMG = "photo_2025-11-21_16-28-17.jpg"

    # Создадим заглушку, если нет картинки, чтобы код не падал
    if not os.path.exists(IMG):
        import cv2

        print("⚠️ Картинка не найдена, создаю тестовую...")
        cv2.imwrite(IMG, np.zeros((512, 512, 3), dtype=np.uint8))

    analyze_rotations(IMG)
