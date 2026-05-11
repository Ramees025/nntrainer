"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>

@file weights_converter.py
@date 08 May 2025
@brief gpt-oss-20b weight converter (binary + safetensors)
@author Eunju Yang <ej.yang@samsung.com>

Weight save order per layer:
  input_layernorm
  self_attn: q_proj (w+b), k_proj (w+b), v_proj (w+b), sinks, o_proj (w+b)
  post_attention_layernorm
  mlp: router (w+b), per-expert: up(w+b), gate(w+b), down(w+b)
Final: model.norm, lm_head
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def save_gpt_oss_for_nntrainer(params, config, dtype, file):
    """Convert and save weights as nntrainer binary format for GPT-OSS."""
    n_layers  = config.num_hidden_layers
    n_experts = config.num_local_experts

    print(dtype)

    def save_weight(weight_name, is_transpose=False):
        if is_transpose:
            print(weight_name, params[weight_name].permute(1, 0).shape,
                  params[weight_name].permute(1, 0).flatten()[:3],
                  params[weight_name].permute(1, 0).flatten()[-3:])
            np.array(params[weight_name].permute(1, 0).float(), dtype=dtype).tofile(file)
        else:
            print(weight_name, params[weight_name].shape,
                  params[weight_name].flatten()[:3],
                  params[weight_name].flatten()[-3:])
            np.array(params[weight_name].float(), dtype=dtype).tofile(file)

    def save_projection(layer_name, proj_name):
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            save_weight(f"{layer_name}{proj_name}.base_layer.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_A.default.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_B.default.weight", True)
        else:
            save_weight(f"{layer_name}{proj_name}.weight", True)

    def save_attention(layer_name):
        for proj in ["q_proj", "k_proj", "v_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            save_weight(f"{layer_name}self_attn.{proj}.bias")
        save_weight(f"{layer_name}self_attn.sinks")
        for proj in ["o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            save_weight(f"{layer_name}self_attn.{proj}.bias")

    def save_feed_forward(layer_name):
        save_weight(f"{layer_name}mlp.router.weight", True)
        save_weight(f"{layer_name}mlp.router.bias")
        for num_expert in range(n_experts):
            up_proj_weight = params[f"{layer_name}mlp.experts.gate_up_proj"][..., 1::2][num_expert]
            up_proj_bias   = params[f"{layer_name}mlp.experts.gate_up_proj_bias"][..., 1::2][num_expert]
            print(f"{layer_name}mlp.experts.gate_up_proj.up",   up_proj_weight.shape)
            print(f"{layer_name}mlp.experts.gate_up_proj.up_b", up_proj_bias.shape)
            np.array(up_proj_weight.float(), dtype=dtype).tofile(file)
            np.array(up_proj_bias.float(),   dtype=dtype).tofile(file)

            gate_proj_weight = params[f"{layer_name}mlp.experts.gate_up_proj"][..., ::2][num_expert]
            gate_proj_bias   = params[f"{layer_name}mlp.experts.gate_up_proj_bias"][..., ::2][num_expert]
            print(f"{layer_name}mlp.experts.gate_up_proj.gate",   gate_proj_weight.shape)
            print(f"{layer_name}mlp.experts.gate_up_proj.gate_b", gate_proj_bias.shape)
            np.array(gate_proj_weight.float(), dtype=dtype).tofile(file)
            np.array(gate_proj_bias.float(),   dtype=dtype).tofile(file)

            down_proj_weight = params[f"{layer_name}mlp.experts.down_proj"][num_expert]
            down_proj_bias   = params[f"{layer_name}mlp.experts.down_proj_bias"][num_expert]
            print(f"{layer_name}mlp.experts.down_proj",   down_proj_weight.shape)
            print(f"{layer_name}mlp.experts.down_proj_b", down_proj_bias.shape)
            np.array(down_proj_weight.float(), dtype=dtype).tofile(file)
            np.array(down_proj_bias.float(),   dtype=dtype).tofile(file)

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


def collect_gpt_oss_for_nntrainer(params, config, dtype):
    """Return list of (nntrainer_name, ndarray) pairs for safetensors output."""
    n_layers  = config.num_hidden_layers
    n_experts = config.num_local_experts
    named = []

    def add(nntr_name, t, transpose=False):
        named.append((nntr_name, _to_np(t, dtype, transpose)))

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

        add(f"{li}_attention_norm:gamma", params[f"{lp}input_layernorm.weight"])

        # Q, K, V with bias
        for proj, nntr in [("q_proj", f"{li}_wq"), ("k_proj", f"{li}_wk"), ("v_proj", f"{li}_wv")]:
            add_proj(nntr, f"{lp}self_attn.{proj}")
            add(f"{nntr}:bias", params[f"{lp}self_attn.{proj}.bias"])

        # Attention sink
        add(f"{li}_attention:sink", params[f"{lp}self_attn.sinks"])

        # O projection with bias
        add_proj(f"{li}_attention_out", f"{lp}self_attn.o_proj")
        add(f"{li}_attention_out:bias", params[f"{lp}self_attn.o_proj.bias"])

        add(f"{li}_ffn_norm:gamma", params[f"{lp}post_attention_layernorm.weight"])

        # MoE layer (gpt_oss_moe type) — gate router
        add(f"{li}_ffn_down:gate",      params[f"{lp}mlp.router.weight"], True)
        add(f"{li}_ffn_down:gate_bias", params[f"{lp}mlp.router.bias"])

        # Per-expert weights (unpacked from gate_up_proj)
        for e in range(n_experts):
            up_w = params[f"{lp}mlp.experts.gate_up_proj"][..., 1::2][e]
            up_b = params[f"{lp}mlp.experts.gate_up_proj_bias"][..., 1::2][e]
            gp_w = params[f"{lp}mlp.experts.gate_up_proj"][..., ::2][e]
            gp_b = params[f"{lp}mlp.experts.gate_up_proj_bias"][..., ::2][e]
            dn_w = params[f"{lp}mlp.experts.down_proj"][e]
            dn_b = params[f"{lp}mlp.experts.down_proj_bias"][e]

            named.append((f"{li}_ffn_down:expert_up_{e}",        _to_np(up_w, dtype)))
            named.append((f"{li}_ffn_down:expert_up_bias_{e}",   _to_np(up_b, dtype)))
            named.append((f"{li}_ffn_down:expert_gate_{e}",      _to_np(gp_w, dtype)))
            named.append((f"{li}_ffn_down:expert_gate_bias_{e}", _to_np(gp_b, dtype)))
            named.append((f"{li}_ffn_down:expert_down_{e}",      _to_np(dn_w, dtype)))
            named.append((f"{li}_ffn_down:expert_down_bias_{e}", _to_np(dn_b, dtype)))

    add("output_norm:gamma",         params["model.norm.weight"])
    add("output_of_causallm:weight", params["lm_head.weight"], True)
    return named


if __name__ == "__main__":
    data_dtype = "float32"
    model_path = "."
    output_name = "./nntr_gpt_oss_20b.bin"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config    = AutoConfig.from_pretrained(model_path)
    model     = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=data_dtype, trust_remote_code=True)
    model.eval()

    if output_name.endswith('.safetensors'):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from safetensors_util import write_safetensors, verify_safetensors
        named = collect_gpt_oss_for_nntrainer(model.state_dict(), config, data_dtype)
        write_safetensors(output_name, named)
        verify_safetensors(output_name)
    else:
        with open(output_name, "wb") as f:
            save_gpt_oss_for_nntrainer(model.state_dict(), config, data_dtype, f)
