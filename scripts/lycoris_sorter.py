import os
from glob import glob

import gradio as gr
from safetensors.torch import load_file
import torch

from modules import script_callbacks


def load_model(path):
    if os.path.splitext(path)[-1] == ".safetensors":
        model = load_file(path)
    else:
        model = torch.load(path)
    return model


def main(input, output):
    yield gr.Button.update(interactive=True)
    try:
        exts = [".safetensors", ".pt", ".ckpt"]

        for file in glob(os.path.join(input, "**", "*"), recursive=True):
            ext = os.path.splitext(file)[-1]
            if ext in exts:
                print(file)
                try:
                    model = load_model(file)
                    print("success")
                except:
                    print("fail")

                if "state_dict" in model:
                    model = model["state_dict"]

                t = "lora"

                for key in model:
                    if "hada" in key:
                        t = "lycoris"

                outdir = os.path.join(output, t)
                outfile = os.path.join(outdir, os.path.relpath(file, input))
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                os.rename(file, outfile)
    except Exception as e:
        print(e)
        yield gr.Button.update(interactive=False)
    yield gr.Button.update(interactive=True)


def on_ui_tabs():
    with gr.Blocks() as ui:
        with gr.Row():
            input = gr.Textbox(label="Input directory")
            output = gr.Textbox(label="Output directory")
        run = gr.Button(label="Run", variant="primary")

        run.click(
            fn=main,
            inputs=[input, output],
            outputs=[run],
        )

    return ((ui, "Lycoris Sorter", "lycoris_sorter"),)


script_callbacks.on_ui_tabs(on_ui_tabs)
