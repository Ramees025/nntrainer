## @file weight_converter.py
## @brief weight conversion script for qwen3_moe model

import argparse
import os
import sys
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def save_qwen3_moe_for_nntrainer(params, config, dtype, file):
    """Convert and save weights as nntrainer binary format for Qwen3-MoE."""
    n_layers  = config.num_hidden_layers
    n_experts = config.num_experts

    def save_weight(weight_name, is_transpose=False):
        print(weight_name, params[weight_name].shape)
        if is_transpose:
            np.array(params[weight_name].permute(1, 0), dtype=dtype).tofile(file)
        else:
            np.array(params[weight_name], dtype=dtype).tofile(file)

    def save_projection(layer_name, proj_name):
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            save_weight(f"{layer_name}{proj_name}.base_layer.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_A.default.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_B.default.weight", True)
        else:
            save_weight(f"{layer_name}{proj_name}.weight", True)

    def save_attention(layer_name):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                save_weight(proj_norm_name)

    def save_feed_forward(layer_name):
        save_weight(f"{layer_name}mlp.gate.weight", True)
        for expert_id in range(n_experts):
            for proj in ["up_proj", "gate_proj", "down_proj"]:
                save_projection(layer_name, f"mlp.experts.{expert_id}.{proj}")

    save_weight("model.embed_tokens.weight")
    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_weight(f"{layer_prefix}input_layernorm.weight")
        save_attention(layer_prefix)
        save_weight(f"{layer_prefix}post_attention_layernorm.weight")
        save_feed_forward(layer_prefix)
    save_weight("model.norm.weight")
    save_weight("lm_head.weight", True)


def _to_np(t, dtype, transpose=False):
    """Convert a PyTorch tensor or numpy array to a numpy array."""
    if hasattr(t, 'detach'):
        arr = t.detach().cpu().float().numpy()
    else:
        arr = np.asarray(t, dtype='float32')
    if transpose and arr.ndim >= 2:
        arr = arr.T
    return np.array(arr, dtype=dtype)


def collect_qwen3_moe_for_nntrainer(params, config, dtype):
    """Return list of (nntrainer_name, ndarray) pairs for safetensors output."""
    n_layers  = config.num_hidden_layers
    n_experts = config.num_experts
    named = []

    def add(nntr_name, t, transpose=False):
        named.append((nntr_name, _to_np(t, dtype, transpose)))

    def add_proj(nntr_layer, hf_key):
        """Add a fully_connected weight (and optional LoRA weights)."""
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

        # MoE layer (qwen_moe type) stored under nntrainer layer "layer{i}_ffn_down".
        # Expert weights use custom roles (expert_up_N, expert_gate_N, expert_down_N),
        # not the standard :weight role, so they must be added directly.
        add(f"{li}_ffn_down:gate", params[f"{lp}mlp.gate.weight"], True)

        for e in range(n_experts):
            ep = f"{lp}mlp.experts.{e}."
            named.append((f"{li}_ffn_down:expert_up_{e}",
                          _to_np(params[f"{ep}up_proj.weight"], dtype, True)))
            named.append((f"{li}_ffn_down:expert_gate_{e}",
                          _to_np(params[f"{ep}gate_proj.weight"], dtype, True)))
            named.append((f"{li}_ffn_down:expert_down_{e}",
                          _to_np(params[f"{ep}down_proj.weight"], dtype, True)))

    add("output_norm:gamma",         params["model.norm.weight"])
    add("output_of_causallm:weight", params["lm_head.weight"], True)
    return named


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, default="./Qwen3-4b")
    parser.add_argument("--output_name", type=str, default="./nntr_qwen3_4b_fp32.bin")
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
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        from safetensors_util import write_safetensors, verify_safetensors
        named = collect_qwen3_moe_for_nntrainer(model.state_dict(), config, data_dtype)
        write_safetensors(output_name, named)
        verify_safetensors(output_name)
    else:
        with open(output_name, "wb") as f:
            save_qwen3_moe_for_nntrainer(model.state_dict(), config, data_dtype, f)
