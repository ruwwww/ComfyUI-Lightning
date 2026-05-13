import sys
import os
from pathlib import Path
from typing import Optional
import types

import torch
import comfy.sd
from comfy.model_patcher import ModelPatcher
from comfy.ldm.anima.model import Anima
from comfy.ldm.cosmos.predict2 import Block, Attention
import comfy.utils
import folder_paths


class ApplySpargeAttnAnima:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "l1": ("FLOAT", {"default": 0.06, "step": 0.0001}),
                "pv_l1": ("FLOAT", {"default": 0.065, "step": 0.0001}),
                "enable_tuning_mode": ("BOOLEAN", {"default": False}),
                "parallel_tuning": ("BOOLEAN", {"default": False}),
                "tuned_hyperparams": (
                    [None] + folder_paths.get_filename_list("checkpoints"),
                    {"default": None},
                ),
                "skip_blocks": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Lightning/SpargeAttn"
    TITLE = "Apply SpargeAttn (Anima)"

    def patch(
        self,
        model: ModelPatcher,
        l1: float,
        pv_l1: float,
        enable_tuning_mode: bool,
        parallel_tuning: bool,
        tuned_hyperparams: Optional[str],
        skip_blocks: str,
    ):
        cloned_model = model.clone()

        try:
            from .utils_anima import (
                make_sparge_attn_op,
                load_sparse_attention_state_dict,
                SparseAttentionMeansim,
            )

            dm: Anima = cloned_model.get_model_object("diffusion_model")
            if isinstance(dm, torch._dynamo.OptimizedModule):
                dm: Anima = getattr(dm, "_orig_mod", dm)

            skip_indices = [
                int(i.strip()) for i in skip_blocks.split(",") if i.strip()
            ]

            for idx, block in enumerate(dm.blocks):
                if idx in skip_indices:
                    continue

                if isinstance(block, torch._dynamo.OptimizedModule):
                    block: Block = getattr(block, "_orig_mod", block)

                # Patch self-attention
                if not hasattr(block.self_attn, "_spargeattn"):
                    sa = SparseAttentionMeansim(l1=l1, pv_l1=pv_l1)
                    block.self_attn.attn_op = make_sparge_attn_op(sa)
                    block.self_attn._spargeattn = sa

                block.self_attn._spargeattn.enable_tuning_mode = enable_tuning_mode

                # Patch cross-attention
                if not hasattr(block.cross_attn, "_spargeattn"):
                    ca = SparseAttentionMeansim(l1=l1, pv_l1=pv_l1)
                    block.cross_attn.attn_op = make_sparge_attn_op(ca)
                    block.cross_attn._spargeattn = ca

                block.cross_attn._spargeattn.enable_tuning_mode = enable_tuning_mode

            if tuned_hyperparams is not None:
                sd_path = folder_paths.get_full_path("checkpoints", tuned_hyperparams)
                sd = comfy.utils.load_torch_file(sd_path, safe_load=True)
                load_sparse_attention_state_dict(dm, sd)

            if parallel_tuning:
                comfyui_root = Path(os.path.abspath(__file__)).resolve().parents[3]
                sys.path.insert(0, str(comfyui_root))
                os.environ["PARALLEL_TUNE"] = "1"
            else:
                os.environ["PARALLEL_TUNE"] = ""

        except Exception as e:
            print(e)

        return (cloned_model,)


class SaveSpargeAttnHyperparamsAnima:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": (
                    "STRING",
                    {"default": "spargeattn_hyperparams_anima"},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "Lightning/SpargeAttn"
    OUTPUT_NODE = True
    TITLE = "Save Finetuned SpargeAttn Hyperparams (Anima)"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    def save(self, model: ModelPatcher, filename_prefix: str):
        try:
            from .utils_anima import extract_sparse_attention_state_dict

            dm: Anima = model.get_model_object("diffusion_model")
            if isinstance(dm, torch._dynamo.OptimizedModule):
                dm: Anima = getattr(dm, "_orig_mod", dm)

            sd = extract_sparse_attention_state_dict(dm)
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(filename_prefix, self.output_dir)
            )
            saved_path = f"{filename}_{counter:05}_.safetensors"
            saved_path = os.path.join(full_output_folder, saved_path)
            comfy.utils.save_torch_file(sd, saved_path, metadata=None)

        except Exception as e:
            print(e)

        return {}


NODE_CLASS_MAPPINGS = {
    "ApplySpargeAttnAnima": ApplySpargeAttnAnima,
    "SaveSpargeAttnHyperparamsAnima": SaveSpargeAttnHyperparamsAnima,
}
