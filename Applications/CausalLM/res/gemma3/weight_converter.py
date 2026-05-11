## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
##
## @file weight_converter.py
## @brief weight conversion script for gemma3 model
## @author SeungBaek Hong <sb92.hong@samsung.com>
##
## nntrainer layer order per decoder block (from gemma3_causallm.cpp):
##   attention_norm, wq, wk, wv, q_norm, k_norm, attention_out,
##   post_attention_norm, pre_ffn_norm (no underscore!),
##   ffn_gate, ffn_up, ffn_down, post_ffn_norm (no underscore!)
## RMSNorm weights: gamma = HF_weight + 1.0  (Gemma3 convention)

import argparse
import os
import sys
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def save_gemma3_for_nntrainer(params, config, dtype, file):
    """Convert and save weights as nntrainer binary format for Gemma3."""
    n_layers = config.num_hidden_layers

    def save_weight(weight, is_rms=False):
        if is_rms:
            weight = weight + 1.0
        np.array(weight, dtype=dtype).tofile(file)

    def save_projection(layer_name, proj_name):
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"].permute(1, 0))
            save_weight(params[f"{layer_name}{proj_name}.lora_A.default.weight"].permute(1, 0))
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"].permute(1, 0))
        else:
            save_weight(params[f"{layer_name}{proj_name}.weight"].permute(1, 0))

    def save_attention(layer_name):
        save_weight(params[f"{layer_name}input_layernorm.weight"], is_rms=True)
        save_projection(layer_name, "self_attn.v_proj")
        save_projection(layer_name, "self_attn.k_proj")
        if f"{layer_name}self_attn.k_norm.weight" in params:
            save_weight(params[f"{layer_name}self_attn.k_norm.weight"], is_rms=True)
        save_projection(layer_name, "self_attn.q_proj")
        if f"{layer_name}self_attn.q_norm.weight" in params:
            save_weight(params[f"{layer_name}self_attn.q_norm.weight"], is_rms=True)
        save_projection(layer_name, "self_attn.o_proj")

    def save_feed_forward(layer_name):
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"], is_rms=True)
        save_weight(params[f"{layer_name}pre_feedforward_layernorm.weight"], is_rms=True)
        for proj in ["up_proj", "gate_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")
        save_weight(params[f"{layer_name}post_feedforward_layernorm.weight"], is_rms=True)

    save_weight(params["model.embed_tokens.weight"])
    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)
    save_weight(params["model.norm.weight"], is_rms=True)
    save_weight(params["lm_head.weight"].permute(1, 0))


def _to_np(t, dtype, transpose=False, rms_add=False):
    """Convert a PyTorch tensor or numpy array to a numpy array."""
    if hasattr(t, 'detach'):
        arr = t.detach().cpu().float().numpy()
    else:
        arr = np.asarray(t, dtype='float32')
    if rms_add:
        arr = arr + 1.0
    if transpose and arr.ndim >= 2:
        arr = arr.T
    return np.array(arr, dtype=dtype)


def collect_gemma3_for_nntrainer(params, config, dtype):
    """Return list of (nntrainer_name, ndarray) pairs for safetensors output.

    Uses nntrainer C++ layer ordering from gemma3_causallm.cpp:
      wq -> wk -> wv -> q_norm -> k_norm -> attention_out
      ffn_gate -> ffn_up -> ffn_down
    Note: layer{i}pre_ffn_norm and layer{i}post_ffn_norm have NO underscore.
    """
    n_layers = config.num_hidden_layers
    named = []

    def add(nntr_name, t, transpose=False, rms_add=False):
        named.append((nntr_name, _to_np(t, dtype, transpose, rms_add)))

    def add_proj(nntr_layer, hf_key):
        lora_key = f"{hf_key}.lora_A.default.weight"
        if lora_key in params:
            add(f"{nntr_layer}:weight",       params[f"{hf_key}.base_layer.weight"], True)
            add(f"{nntr_layer}:lora_A_weight", params[lora_key], True)
            add(f"{nntr_layer}:lora_B_weight", params[f"{hf_key}.lora_B.default.weight"], True)
        else:
            add(f"{nntr_layer}:weight", params[f"{hf_key}.weight"], True)

    add("embedding0:weight", params["model.embed_tokens.weight"])

    for i in range(n_layers):
        lp = f"model.layers.{i}."  # HF layer prefix
        li = f"layer{i}"           # nntrainer layer prefix

        # Attention norm
        add(f"{li}_attention_norm:gamma", params[f"{lp}input_layernorm.weight"], rms_add=True)

        # Attention: Q -> K -> V -> q_norm -> k_norm -> O  (gemma3_causallm.cpp order)
        add_proj(f"{li}_wq", f"{lp}self_attn.q_proj")
        add_proj(f"{li}_wk", f"{lp}self_attn.k_proj")
        add_proj(f"{li}_wv", f"{lp}self_attn.v_proj")
        if f"{lp}self_attn.q_norm.weight" in params:
            add(f"{li}_q_norm:gamma", params[f"{lp}self_attn.q_norm.weight"], rms_add=True)
        if f"{lp}self_attn.k_norm.weight" in params:
            add(f"{li}_k_norm:gamma", params[f"{lp}self_attn.k_norm.weight"], rms_add=True)
        add_proj(f"{li}_attention_out", f"{lp}self_attn.o_proj")

        # Post-attention norm
        add(f"{li}_post_attention_norm:gamma",
            params[f"{lp}post_attention_layernorm.weight"], rms_add=True)

        # Pre-FFN norm  — layer name has NO underscore before 'pre'
        add(f"{li}pre_ffn_norm:gamma",
            params[f"{lp}pre_feedforward_layernorm.weight"], rms_add=True)

        # FFN: gate -> (gelu, no weight) -> up -> (multiply, no weight) -> down
        add_proj(f"{li}_ffn_gate", f"{lp}mlp.gate_proj")
        add_proj(f"{li}_ffn_up",   f"{lp}mlp.up_proj")
        add_proj(f"{li}_ffn_down", f"{lp}mlp.down_proj")

        # Post-FFN norm  — layer name has NO underscore before 'post'
        add(f"{li}post_ffn_norm:gamma",
            params[f"{lp}post_feedforward_layernorm.weight"], rms_add=True)

    add("output_norm:gamma",         params["model.norm.weight"], rms_add=True)
    add("output_of_causallm:weight", params["lm_head.weight"], True)
    return named


if __name__ == "__main__":
    data_dtype  = "float32"
    model_path  = "./270m"
    output_name = "./nntr_gemma3_270m_fp32.bin"

    config = AutoConfig.from_pretrained(model_path)
    model  = AutoModelForCausalLM.from_pretrained(model_path, dtype="float", trust_remote_code=True)
    model.eval()

    print(model)

    if output_name.endswith('.safetensors'):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from safetensors_util import write_safetensors, verify_safetensors
        named = collect_gemma3_for_nntrainer(model.state_dict(), config, data_dtype)
        write_safetensors(output_name, named)
        verify_safetensors(output_name)
    else:
        with open(output_name, "wb") as f:
            save_gemma3_for_nntrainer(model.state_dict(), config, data_dtype, f)
