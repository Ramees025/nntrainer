## @file weight_converter.py
## @brief weight conversion script for qwen3-0.6b model
## @author Eunju Yang <ej.yang@samsung.com>

import argparse
import os
import sys
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def save_qwen3_for_nntrainer(params, n_layers, dtype, file):
    """Convert and save weights as nntrainer binary format for Qwen3."""

    def save_weight(weight):
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
        save_weight(params[f"{layer_name}input_layernorm.weight"])
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                save_weight(params[proj_norm_name])

    def save_feed_forward(layer_name):
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"])
        for proj in ["up_proj", "gate_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")

    save_weight(params["model.embed_tokens.weight"])
    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)
    save_weight(params["model.norm.weight"])
    save_weight(params["lm_head.weight"].permute(1, 0))


def _to_np(t, dtype, transpose=False):
    """Convert a PyTorch tensor or numpy array to a numpy array."""
    if hasattr(t, 'detach'):
        arr = t.detach().cpu().float().numpy()
    else:
        arr = np.asarray(t, dtype='float32')
    if transpose and arr.ndim >= 2:
        arr = arr.T
    return np.array(arr, dtype=dtype)


def collect_qwen3_for_nntrainer(params, n_layers, dtype):
    """Return list of (nntrainer_name, ndarray) pairs for safetensors output."""
    named = []

    def add(nntr_name, t, transpose=False):
        named.append((nntr_name, _to_np(t, dtype, transpose)))

    def add_proj(nntr_layer, hf_key):
        lora_key = f"{hf_key}.lora_A.default.weight"
        if lora_key in params:
            add(f"{nntr_layer}:weight",        params[f"{hf_key}.base_layer.weight"], True)
            add(f"{nntr_layer}:lora_A_weight", params[lora_key], True)
            add(f"{nntr_layer}:lora_B_weight", params[f"{hf_key}.lora_B.default.weight"], True)
        else:
            add(f"{nntr_layer}:weight", params[f"{hf_key}.weight"], True)

    add("embedding0:weight", params["model.embed_tokens.weight"])

    for i in range(n_layers):
        lp = f"model.layers.{i}."  # HF layer prefix
        li = f"layer{i}"           # nntrainer layer prefix

        add(f"{li}_attention_norm:gamma", params[f"{lp}input_layernorm.weight"])

        # Q -> q_norm -> K -> k_norm -> V -> O  (Qwen3 attention order)
        add_proj(f"{li}_wq", f"{lp}self_attn.q_proj")
        if f"{lp}self_attn.q_norm.weight" in params:
            add(f"{li}_q_norm:gamma", params[f"{lp}self_attn.q_norm.weight"])
        add_proj(f"{li}_wk", f"{lp}self_attn.k_proj")
        if f"{lp}self_attn.k_norm.weight" in params:
            add(f"{li}_k_norm:gamma", params[f"{lp}self_attn.k_norm.weight"])
        add_proj(f"{li}_wv",            f"{lp}self_attn.v_proj")
        add_proj(f"{li}_attention_out", f"{lp}self_attn.o_proj")

        add(f"{li}_ffn_norm:gamma", params[f"{lp}post_attention_layernorm.weight"])
        add_proj(f"{li}_ffn_up",   f"{lp}mlp.up_proj")
        add_proj(f"{li}_ffn_gate", f"{lp}mlp.gate_proj")
        add_proj(f"{li}_ffn_down", f"{lp}mlp.down_proj")

    add("output_norm:gamma",         params["model.norm.weight"])
    add("output_of_causallm:weight", params["lm_head.weight"], True)
    return named


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_name", type=str, default="./nntr_qwen3_0.6b_fp32.bin")
    parser.add_argument("--data_type",   type=str, default="float32")
    args = parser.parse_args()

    data_dtype  = args.data_type
    model_path  = args.model_path
    output_name = args.output_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config    = AutoConfig.from_pretrained(model_path)
    model     = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="float", trust_remote_code=True)
    model.eval()

    if output_name.endswith('.safetensors'):
        # res/ is two levels up from res/qwen3/qwen3-0.6b/
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from safetensors_util import write_safetensors, verify_safetensors
        named = collect_qwen3_for_nntrainer(model.state_dict(), config.num_hidden_layers, data_dtype)
        write_safetensors(output_name, named)
        verify_safetensors(output_name)
    else:
        with open(output_name, "wb") as f:
            save_qwen3_for_nntrainer(model.state_dict(), config.num_hidden_layers, data_dtype, f)
