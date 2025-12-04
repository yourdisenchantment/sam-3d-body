#!/usr/bin/env python3
"""
generate_config.py
Конфигурация "Все точки":
- Sapiens 308 (Поверхность)
- MHR 127 (Внутренности)
- Без костей (Edges = [])
"""


def generate_config():
    lines = []
    lines.append("# config_skeleton.py")
    lines.append("# ALL POINTS CONFIGURATION")
    lines.append("")

    # =========================================================
    # 1. MHR 127 (ВНУТРЕННИЙ СКЕЛЕТ)
    # =========================================================
    MHR_NAMES = {
        0: "MHR_Pelvis",
        1: "MHR_Spine1",
        2: "MHR_Spine2",
        3: "MHR_Spine3",
        4: "MHR_Spine4",
        5: "MHR_Spine5",
        6: "MHR_Spine6",
        7: "MHR_Spine7",
        8: "MHR_Spine8",
        9: "MHR_Spine9",
        10: "MHR_Spine10",
        11: "MHR_Spine11",
        12: "MHR_Neck",
        13: "MHR_Head",
        14: "MHR_Jaw",
        15: "MHR_L_Eye",
        16: "MHR_R_Eye",
        17: "MHR_L_Ear",
        18: "MHR_R_Ear",
        19: "MHR_L_Clavicle",
        20: "MHR_L_Shoulder",
        21: "MHR_L_Elbow",
        22: "MHR_L_Wrist",
        23: "MHR_L_IndexMeta",
        24: "MHR_L_MiddleMeta",
        25: "MHR_L_RingMeta",
        26: "MHR_L_PinkyMeta",
        27: "MHR_L_ThumbMeta",
        28: "MHR_R_Clavicle",
        29: "MHR_R_Shoulder",
        30: "MHR_R_Elbow",
        31: "MHR_R_Wrist",
        32: "MHR_R_IndexMeta",
        33: "MHR_R_MiddleMeta",
        34: "MHR_R_RingMeta",
        35: "MHR_R_PinkyMeta",
        36: "MHR_R_ThumbMeta",
        37: "MHR_L_Hip",
        38: "MHR_L_Knee",
        39: "MHR_L_Ankle",
        40: "MHR_L_Subtalar",
        41: "MHR_L_Meta1",
        42: "MHR_L_Meta2",
        43: "MHR_L_Meta3",
        44: "MHR_L_Meta4",
        45: "MHR_L_Meta5",
        46: "MHR_R_Hip",
        47: "MHR_R_Knee",
        48: "MHR_R_Ankle",
        49: "MHR_R_Subtalar",
        50: "MHR_R_Meta1",
        51: "MHR_R_Meta2",
        52: "MHR_R_Meta3",
        53: "MHR_R_Meta4",
        54: "MHR_R_Meta5",
    }
    # Добиваем пальцы рук и ног MHR (55-126)
    for i in range(55, 127):
        MHR_NAMES[i] = f"MHR_Joint_{i}"

    # =========================================================
    # 2. SAPIENS 308 (ПОВЕРХНОСТЬ)
    # =========================================================
    SAPIENS_NAMES = {
        0: "SAP_Nose",
        1: "SAP_L_Eye",
        2: "SAP_R_Eye",
        3: "SAP_L_Ear",
        4: "SAP_R_Ear",
        5: "SAP_L_Shoulder",
        6: "SAP_R_Shoulder",
        7: "SAP_L_Elbow",
        8: "SAP_R_Elbow",
        9: "SAP_L_Wrist",
        10: "SAP_R_Wrist",
        11: "SAP_L_Hip",
        12: "SAP_R_Hip",
        13: "SAP_L_Knee",
        14: "SAP_R_Knee",
        15: "SAP_L_Ankle",
        16: "SAP_R_Ankle",
        17: "SAP_L_BigToe",
        18: "SAP_L_SmallToe",
        19: "SAP_L_Heel",
        20: "SAP_R_BigToe",
        21: "SAP_R_SmallToe",
        22: "SAP_R_Heel",
    }
    # Руки (23-64)
    for i in range(23, 65):
        SAPIENS_NAMES[i] = f"SAP_Hand_{i}"
    # Лицо (65-307)
    for i in range(65, 308):
        SAPIENS_NAMES[i] = f"SAP_Face_{i}"

    # =========================================================
    # 3. ГРУППЫ ИНДЕКСОВ
    # =========================================================

    # MHR (Все 127)
    MHR_INDICES = list(MHR_NAMES.keys())

    # Sapiens Body (0-64)
    SAPIENS_BODY_INDICES = list(range(0, 65))

    # Sapiens Face (65-307)
    SAPIENS_FACE_INDICES = list(range(65, 308))

    # =========================================================
    # ЗАПИСЬ
    # =========================================================
    lines.append("# --- СЛОВАРИ ИМЕН ---")
    lines.append(f"MHR_NAMES = {MHR_NAMES}")
    lines.append(f"SAPIENS_NAMES = {SAPIENS_NAMES}")
    lines.append("")

    lines.append("# --- ГРУППЫ ИНДЕКСОВ ---")
    lines.append(f"MHR_INDICES = {MHR_INDICES}")
    lines.append(f"SAPIENS_BODY_INDICES = {SAPIENS_BODY_INDICES}")
    lines.append(f"SAPIENS_FACE_INDICES = {SAPIENS_FACE_INDICES}")
    lines.append("")

    lines.append("# --- СВЯЗКИ (ПОКА ПУСТО) ---")
    lines.append("BONES = []")
    lines.append("")

    lines.append("# --- СТИЛИ (ЦВЕТА) ---")
    lines.append("STYLE = {")
    lines.append("    # MHR (Внутренний скелет) - Красный")
    lines.append("    'mhr_color': [255, 0, 0, 255],")
    lines.append("    'mhr_radius': 0.008,")
    lines.append("")
    lines.append("    # Sapiens Body (Поверхность) - Синий")
    lines.append("    'sap_body_color': [0, 0, 255, 255],")
    lines.append("    'sap_body_radius': 0.008,")
    lines.append("")
    lines.append("    # Sapiens Face (Мимика) - Желтый")
    lines.append("    'sap_face_color': [255, 255, 0, 255],")
    lines.append("    'sap_face_radius': 0.003,")
    lines.append("}")

    with open("config_skeleton.py", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ config_skeleton.py создан.")
    print(f"   MHR точек: {len(MHR_NAMES)}")
    print(f"   Sapiens точек: {len(SAPIENS_NAMES)}")
    print("   Связи (Bones): 0 (отключены)")


if __name__ == "__main__":
    generate_config()
