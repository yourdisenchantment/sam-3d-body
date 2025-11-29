import gradio as gr
import os
import json
import uuid
from pathlib import Path
from core import BodyReconstructor

# Инициализация
reconstructor = BodyReconstructor()


def run_inference(image):
    if image is None:
        return None, None, None

    # Папка для логов
    uid = uuid.uuid4().hex[:6]
    save_dir = Path(f"output/gradio_{uid}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Обработка
    res = reconstructor.process(image, output_dir=str(save_dir))

    if not res["scene_full"]:
        return None, None, None

    # Сохраняем 3 варианта
    path_full = save_dir / "full.glb"
    path_body = save_dir / "body.glb"
    path_skel = save_dir / "skel.glb"

    res["scene_full"].export(path_full)
    if res["scene_body"]:
        res["scene_body"].export(path_body)
    if res["scene_skel"]:
        res["scene_skel"].export(path_skel)

    # JSON
    path_json = save_dir / "data.json"
    with open(path_json, "w") as f:
        json.dump(res["json_data"], f, indent=2)

    # Возвращаем словарь путей для State и путь FULL для отображения
    paths = {
        "full": str(path_full.absolute()),
        "body": str(path_body.absolute()),
        "skel": str(path_skel.absolute()),
    }

    return paths, str(path_json), str(path_full.absolute())


# Функция переключения вида
def change_view(mode, paths):
    if not paths:
        return None
    if mode == "Full":
        return paths["full"]
    if mode == "Body Only":
        return paths["body"]
    if mode == "Skeleton Only":
        return paths["skel"]
    return paths["full"]


# --- UI ---
with gr.Blocks(title="SAM 3D Body") as demo:
    gr.Markdown("### 🧍 SAM 3D Body")

    # Хранилище путей (скрытое)
    paths_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="filepath", label="Input")
            btn_gen = gr.Button("🚀 GENERATE", variant="primary")

            gr.Markdown("### View Controls")
            with gr.Row():
                btn_full = gr.Button("Full")
                btn_body = gr.Button("Body")
                btn_skel = gr.Button("Skeleton")

        with gr.Column(scale=2):
            out_3d = gr.Model3D(
                label="3D Result", clear_color=[0.9, 0.9, 0.9, 1.0], interactive=True
            )
            out_json = gr.File(label="JSON Output")

    # Логика
    btn_gen.click(run_inference, inputs=inp, outputs=[paths_state, out_json, out_3d])

    btn_full.click(lambda p: change_view("Full", p), inputs=paths_state, outputs=out_3d)
    btn_body.click(
        lambda p: change_view("Body Only", p), inputs=paths_state, outputs=out_3d
    )
    btn_skel.click(
        lambda p: change_view("Skeleton Only", p), inputs=paths_state, outputs=out_3d
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
