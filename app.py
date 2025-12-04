import gradio as gr
import json
import uuid
import trimesh
import trimesh.util
from pathlib import Path
from core import BodyReconstructor

reconstructor = BodyReconstructor()


def run_process(image):
    if image is None:
        return None, None, None, None, None

    uid = uuid.uuid4().hex[:6]
    save_dir = Path(f"output/gradio_{uid}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Генерация
    res = reconstructor.process(image, output_dir=str(save_dir))

    # 2. Сохранение JSON (Словарь имен)
    json_path = save_dir / "skeleton_data.json"
    with open(json_path, "w") as f:
        json.dump(res["json_data"], f, indent=2)

    # 3. Сохранение GLB Скелета (Все точки)
    skel_meshes = (
        res["raw_meshes"]["mhr"]
        + res["raw_meshes"]["sap_body"]
        + res["raw_meshes"]["sap_face"]
    )
    skel_path = None
    if skel_meshes:
        skel_scene = trimesh.Scene(skel_meshes)
        skel_path = save_dir / "skeleton_points.glb"
        skel_scene.export(skel_path)

    # 4. Сохранение GLB Мэша (Тело)
    mesh_path = None
    if res["raw_meshes"]["body"]:
        body_scene = trimesh.Scene(res["raw_meshes"]["body"])
        mesh_path = save_dir / "body_mesh.glb"
        body_scene.export(mesh_path)

    # Возвращаем: StateRes, StateDir, PathJSON, PathSkelGLB, PathMeshGLB
    return (
        res,
        str(save_dir),
        str(json_path),
        str(skel_path) if skel_path else None,
        str(mesh_path) if mesh_path else None,
    )


def update_visuals(res_state, save_dir, filters):
    if not res_state or not save_dir:
        return None

    scene_objs = []
    raw = res_state["raw_meshes"]

    # Фильтры
    if "Mesh (Skin)" in filters:
        scene_objs.extend(raw["body"])

    if "MHR Skeleton (Internal)" in filters:
        scene_objs.extend(raw["mhr"])

    if "Sapiens Body (Surface)" in filters:
        scene_objs.extend(raw["sap_body"])

    if "Sapiens Face (Mimicry)" in filters:
        scene_objs.extend(raw["sap_face"])

    if not scene_objs:
        return None

    # Экспорт временного файла для вьювера
    scene = trimesh.Scene(scene_objs)
    out_name = f"view_{uuid.uuid4().hex[:4]}.glb"
    out_path = Path(save_dir) / out_name
    scene.export(out_path)

    return str(out_path.absolute())


# --- UI LAYOUT ---
with gr.Blocks(title="SAM 3D Analysis Tool") as demo:
    state_res = gr.State()
    state_dir = gr.State()

    with gr.Row():
        # --- LEFT SIDEBAR ---
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("## 🛠️ Control Panel")

            # Input
            inp_img = gr.Image(type="filepath", label="Upload Image")
            btn_gen = gr.Button("▶️ GENERATE", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### 👁️ View Filters")

            # Filters
            check_view = gr.CheckboxGroup(
                choices=[
                    "Mesh (Skin)",
                    "MHR Skeleton (Internal)",
                    "Sapiens Body (Surface)",
                    "Sapiens Face (Mimicry)",
                ],
                value=["MHR Skeleton (Internal)", "Sapiens Body (Surface)"],
                label="Show/Hide Elements",
            )

            gr.Markdown("---")
            gr.Markdown("### 💾 Downloads")

            # Download Buttons
            with gr.Row():
                file_json = gr.File(label="Skeleton Data (JSON)")
            with gr.Row():
                file_skel = gr.File(label="Skeleton 3D (GLB)")
                file_mesh = gr.File(label="Body Mesh 3D (GLB)")

        # --- RIGHT MAIN VIEW ---
        with gr.Column(scale=4):
            gr.Markdown("## 3D Viewer")
            out_viewer = gr.Model3D(
                label="Result", clear_color=[0.1, 0.1, 0.1, 1.0], interactive=True
            )

    # --- LOGIC ---

    # 1. Generate -> Save Files -> Update View
    btn_gen.click(
        fn=run_process,
        inputs=[inp_img],
        outputs=[state_res, state_dir, file_json, file_skel, file_mesh],
    ).success(
        fn=update_visuals,
        inputs=[state_res, state_dir, check_view],
        outputs=[out_viewer],
    )

    # 2. Change Filters -> Update View
    check_view.change(
        fn=update_visuals,
        inputs=[state_res, state_dir, check_view],
        outputs=[out_viewer],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
