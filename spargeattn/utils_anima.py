import torch
from torch import Tensor, nn

from spas_sage_attn.autotune import (
    SparseAttentionMeansim,
    extract_sparse_attention_state_dict,
)


def load_sparse_attention_state_dict(model, saved_state_dict, verbose=False):
    device = next(model.parameters()).device
    loaded_count = 0

    for k, v in model.named_modules():
        if isinstance(
            v, SparseAttentionMeansim
        ):  # find each SparseAttentionMeansim instance
            if verbose:
                print(
                    k, "is an instance of SparseAttentionMeansim, but it is empty now."
                )
            for sk, sv in saved_state_dict.items():
                if k in sk:
                    if verbose:
                        print(f"{sk} is a substate_dict of {k}, we will load it.")

                    sub_name = sk.split(k)[1][1:]
                    sv = sv.to(device=device)
                    setattr(v, sub_name, nn.Parameter(sv, requires_grad=False))
                    loaded_count += 1
                    
    if loaded_count == 0 and len(saved_state_dict) > 0:
        raise ValueError(
            "SpargeAttn failed to load any hyperparameter keys from the provided state dict. "
            "This usually happens if you try to load hyperparameters generated for a different model architecture "
            "(e.g., trying to use Flux hyperparameters on an Anima model). "
            "Please run the tuning process specifically for this model architecture first."
        )
    return model


def make_sparge_attn_op(spargeattn: SparseAttentionMeansim):
    """Create a sparge attention op matching predict2's torch_attention_op signature.

    The predict2 Attention class calls:
        self.attn_op(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D, transformer_options={})

    where tensors have shape (B, S, H, D) — "NHD" layout in sparge terms.
    The return value must be shape (B, S, H*D) — flattened heads, ready for output_proj.
    """
    def sparge_attn_op(
        q_B_S_H_D: Tensor,
        k_B_S_H_D: Tensor,
        v_B_S_H_D: Tensor,
        transformer_options=None,
    ) -> Tensor:
        tune_mode = getattr(spargeattn, "enable_tuning_mode", False)
        
        if not tune_mode and spargeattn.cdfthreshd is None:
            import os
            if os.environ.get("TUNE_MODE", "") == "":
                raise RuntimeError(
                    "SpargeAttn error: 'attention hyperparameters should be tuned first'. "
                    "You are running in inference mode but no valid tuned hyperparameters were loaded. "
                    "Because Anima has a different architecture than Flux, you cannot use Flux hyperparameters. "
                    "Please either check 'enable_tuning_mode' to tune the Anima model, or provide valid Anima 'tuned_hyperparams'."
                )

        # SparseAttentionMeansim accepts NHD layout: (B, S, H, D)
        out = spargeattn(
            q_B_S_H_D,
            k_B_S_H_D,
            v_B_S_H_D,
            mask=None,
            is_causal=False,
            tensor_layout="NHD",
            tune_mode=tune_mode,
            return_sparsity=False,
        )

        # Flatten heads → (B, S, H*D) to match output_proj input
        B, S = out.shape[:2]
        return out.reshape(B, S, -1)

    return sparge_attn_op
