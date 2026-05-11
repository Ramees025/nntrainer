# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics
#
# @file  test_weight_converter.py
# @brief Validation tests for CausalLM safetensors weight converters.
#
# Tests run WITHOUT actual model weights (uses synthetic numpy tensors).
# Run with:  python Applications/CausalLM/res/test_weight_converter.py

import json
import os
import struct
import sys
import tempfile
import traceback

import numpy as np

RES_DIR = os.path.dirname(__file__)
sys.path.insert(0, RES_DIR)
from safetensors_util import write_safetensors, verify_safetensors


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rng(shape, dtype='float32'):
    return np.random.default_rng(42).random(shape).astype(dtype)


def _read_header(path):
    with open(path, 'rb') as f:
        hsize = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hsize).decode('utf-8'))
        data   = f.read()
    return header, data


PASSED = []
FAILED = []


def run(fn):
    name = fn.__name__
    try:
        fn()
        print(f'  [PASS] {name}')
        PASSED.append(name)
    except Exception as e:
        print(f'  [FAIL] {name}: {e}')
        traceback.print_exc()
        FAILED.append(name)


# ---------------------------------------------------------------------------
# format tests
# ---------------------------------------------------------------------------

def test_header_is_8byte_aligned():
    tensors = [("a:b", _rng((1,)))]
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        path = f.name
    try:
        write_safetensors(path, tensors)
        header, _ = _read_header(path)
        with open(path, 'rb') as fh:
            hsize = struct.unpack('<Q', fh.read(8))[0]
        assert hsize % 8 == 0, f"header size {hsize} not 8-byte aligned"
    finally:
        os.unlink(path)


def test_metadata_format_field():
    tensors = [("layer0_wq:weight", _rng((8, 8)))]
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        path = f.name
    try:
        write_safetensors(path, tensors)
        header, _ = _read_header(path)
        assert header['__metadata__']['format'] == 'nntrainer'
    finally:
        os.unlink(path)


def test_roundtrip_values():
    tensors = [
        ("embedding0:weight",          _rng((32, 16))),
        ("layer0_attention_norm:gamma", _rng((16,))),
        ("layer0_wq:weight",            _rng((16, 16))),
        ("output_norm:gamma",           _rng((16,))),
    ]
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        path = f.name
    try:
        write_safetensors(path, tensors)
        header, data = _read_header(path)
        for name, expected in tensors:
            assert name in header, f"missing tensor {name}"
            info  = header[name]
            s, e  = info['data_offsets']
            actual = np.frombuffer(data[s:e], dtype='float32').reshape(expected.shape)
            assert np.allclose(actual, expected), f"data mismatch for {name}"
    finally:
        os.unlink(path)


def test_multiple_dtypes():
    tensors = [
        ("fp32_tensor", _rng((4, 4), 'float32')),
        ("fp16_tensor", _rng((4, 4), 'float16')),
    ]
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        path = f.name
    try:
        write_safetensors(path, tensors)
        header, _ = _read_header(path)
        assert header['fp32_tensor']['dtype'] == 'F32'
        assert header['fp16_tensor']['dtype'] == 'F16'
    finally:
        os.unlink(path)


def test_data_offsets_contiguous():
    """data_offsets of consecutive tensors must be contiguous (no gaps)."""
    tensors = [(f"t{i}:w", _rng((4,))) for i in range(5)]
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        path = f.name
    try:
        write_safetensors(path, tensors)
        header, _ = _read_header(path)
        names = [n for n, _ in tensors]
        prev_end = 0
        for name in names:
            s, e = header[name]['data_offsets']
            assert s == prev_end, f"{name}: expected start={prev_end}, got {s}"
            prev_end = e
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# nntrainer name mapping tests (using synthetic params dicts)
# ---------------------------------------------------------------------------

def _make_qwen2_params(n_layers=2, hidden=8, inter=16, heads=2, head_dim=4, vocab=32):
    """Build a minimal synthetic params dict matching Qwen2 state_dict structure."""
    p = {}
    p['model.embed_tokens.weight'] = _rng((vocab, hidden))
    for i in range(n_layers):
        lp = f'model.layers.{i}.'
        p[f'{lp}input_layernorm.weight']            = _rng((hidden,))
        p[f'{lp}self_attn.q_proj.weight']           = _rng((heads * head_dim, hidden))
        p[f'{lp}self_attn.q_proj.bias']             = _rng((heads * head_dim,))
        p[f'{lp}self_attn.k_proj.weight']           = _rng((heads * head_dim, hidden))
        p[f'{lp}self_attn.k_proj.bias']             = _rng((heads * head_dim,))
        p[f'{lp}self_attn.v_proj.weight']           = _rng((heads * head_dim, hidden))
        p[f'{lp}self_attn.v_proj.bias']             = _rng((heads * head_dim,))
        p[f'{lp}self_attn.o_proj.weight']           = _rng((hidden, heads * head_dim))
        p[f'{lp}post_attention_layernorm.weight']   = _rng((hidden,))
        p[f'{lp}mlp.up_proj.weight']                = _rng((inter, hidden))
        p[f'{lp}mlp.gate_proj.weight']              = _rng((inter, hidden))
        p[f'{lp}mlp.down_proj.weight']              = _rng((hidden, inter))
    p['model.norm.weight'] = _rng((hidden,))
    return p


def test_qwen2_names():
    sys.path.insert(0, os.path.join(RES_DIR, 'qwen2', 'qwen2-0.5b'))
    from weight_converter import collect_qwen2_for_nntrainer
    N = 2
    params = _make_qwen2_params(n_layers=N)
    named  = collect_qwen2_for_nntrainer(params, N, 'float32')
    name_set = {n for n, _ in named}

    assert 'embedding0:weight' in name_set
    assert 'output_norm:gamma' in name_set
    assert 'output_of_causallm:weight' not in name_set, 'Qwen2 is tie_word_embeddings'
    for i in range(N):
        li = f'layer{i}'
        for n in [f'{li}_attention_norm:gamma',
                  f'{li}_wq:weight', f'{li}_wq:bias',
                  f'{li}_wk:weight', f'{li}_wk:bias',
                  f'{li}_wv:weight', f'{li}_wv:bias',
                  f'{li}_attention_out:weight',
                  f'{li}_ffn_norm:gamma',
                  f'{li}_ffn_up:weight', f'{li}_ffn_gate:weight', f'{li}_ffn_down:weight']:
            assert n in name_set, f'missing {n}'

    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        path = f.name
    try:
        write_safetensors(path, named)
        h = verify_safetensors(path)
        assert h['__metadata__']['format'] == 'nntrainer'
    finally:
        os.unlink(path)


def _make_qwen3_params(n_layers=2, hidden=8, inter=16, heads=2, head_dim=4, vocab=32):
    p = _make_qwen2_params(n_layers, hidden, inter, heads, head_dim, vocab)
    for i in range(n_layers):
        lp = f'model.layers.{i}.'
        # Remove Qwen2 biases
        for proj in ['q_proj', 'k_proj', 'v_proj']:
            del p[f'{lp}self_attn.{proj}.bias']
        # Add q_norm, k_norm
        p[f'{lp}self_attn.q_norm.weight'] = _rng((head_dim,))
        p[f'{lp}self_attn.k_norm.weight'] = _rng((head_dim,))
    p['lm_head.weight'] = _rng((vocab, hidden))
    return p


def test_qwen3_names():
    sys.path.insert(0, os.path.join(RES_DIR, 'qwen3', 'qwen3-4b'))
    from weight_converter import collect_qwen3_for_nntrainer
    N = 2
    params = _make_qwen3_params(n_layers=N)
    named  = collect_qwen3_for_nntrainer(params, N, 'float32')
    name_set = {n for n, _ in named}

    assert 'output_of_causallm:weight' in name_set, 'Qwen3 has separate lm_head'
    for i in range(N):
        li = f'layer{i}'
        for n in [f'{li}_wq:weight', f'{li}_q_norm:gamma',
                  f'{li}_wk:weight', f'{li}_k_norm:gamma',
                  f'{li}_wv:weight', f'{li}_attention_out:weight']:
            assert n in name_set, f'missing {n}'


def _make_gemma3_params(n_layers=2, hidden=8, inter=16, heads=2, head_dim=4, vocab=32):
    p = {}
    p['model.embed_tokens.weight'] = _rng((vocab, hidden))
    for i in range(n_layers):
        lp = f'model.layers.{i}.'
        p[f'{lp}input_layernorm.weight']          = _rng((hidden,))
        p[f'{lp}self_attn.q_proj.weight']         = _rng((heads * head_dim, hidden))
        p[f'{lp}self_attn.k_proj.weight']         = _rng((heads * head_dim, hidden))
        p[f'{lp}self_attn.v_proj.weight']         = _rng((heads * head_dim, hidden))
        p[f'{lp}self_attn.q_norm.weight']         = _rng((head_dim,))
        p[f'{lp}self_attn.k_norm.weight']         = _rng((head_dim,))
        p[f'{lp}self_attn.o_proj.weight']         = _rng((hidden, heads * head_dim))
        p[f'{lp}post_attention_layernorm.weight'] = _rng((hidden,))
        p[f'{lp}pre_feedforward_layernorm.weight']  = _rng((hidden,))
        p[f'{lp}mlp.gate_proj.weight']            = _rng((inter, hidden))
        p[f'{lp}mlp.up_proj.weight']              = _rng((inter, hidden))
        p[f'{lp}mlp.down_proj.weight']            = _rng((hidden, inter))
        p[f'{lp}post_feedforward_layernorm.weight'] = _rng((hidden,))
    p['model.norm.weight'] = _rng((hidden,))
    p['lm_head.weight']    = _rng((vocab, hidden))
    return p


def test_gemma3_names():
    sys.path.insert(0, os.path.join(RES_DIR, 'gemma3'))
    from weight_converter import collect_gemma3_for_nntrainer

    class MockConfig:
        num_hidden_layers = 2

    params = _make_gemma3_params(n_layers=2)
    named  = collect_gemma3_for_nntrainer(params, MockConfig(), 'float32')
    name_set = {n for n, _ in named}

    for i in range(2):
        li = f'layer{i}'
        for n in [f'{li}_attention_norm:gamma',
                  f'{li}_wq:weight', f'{li}_wk:weight', f'{li}_wv:weight',
                  f'{li}_q_norm:gamma', f'{li}_k_norm:gamma',
                  f'{li}_attention_out:weight',
                  f'{li}_post_attention_norm:gamma',
                  f'{li}pre_ffn_norm:gamma',     # NO underscore!
                  f'{li}_ffn_gate:weight', f'{li}_ffn_up:weight', f'{li}_ffn_down:weight',
                  f'{li}post_ffn_norm:gamma']:
            assert n in name_set, f'missing Gemma3 tensor {n}'

    # Verify RMSNorm +1 offset is applied
    for name, arr in named:
        if ':gamma' in name:
            i = [n for n, _ in named].index(name)
            # gamma arrays should all be > 0 (rng() is in [0,1), +1 makes it [1,2))
            assert np.all(arr >= 1.0), f'{name} gamma not offset by +1'
            break


def test_kalm_embedding_names():
    sys.path.insert(0, os.path.join(RES_DIR, 'kalm-embedding'))
    from weight_converter import collect_kalm_embedding_for_nntrainer
    N = 2
    # Build params with 0.auto_model. prefix
    base = _make_qwen2_params(n_layers=N)
    params = {}
    params['0.auto_model.embed_tokens.weight'] = base['model.embed_tokens.weight']
    params['0.auto_model.norm.weight']         = base['model.norm.weight']
    for i in range(N):
        src = f'model.layers.{i}.'
        dst = f'0.auto_model.layers.{i}.'
        for k, v in base.items():
            if k.startswith(src):
                params[k.replace(src, dst)] = v

    named    = collect_kalm_embedding_for_nntrainer(params, N, 'float32')
    name_set = {n for n, _ in named}
    assert 'embedding0:weight' in name_set
    assert 'output_norm:gamma' in name_set
    assert 'output_of_causallm:weight' not in name_set, 'KALM is embedding-only'


# ---------------------------------------------------------------------------
# verify_safetensors smoke test
# ---------------------------------------------------------------------------

def test_verify_safetensors():
    tensors = [(f'layer{i}_wq:weight', _rng((8, 8))) for i in range(3)]
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        path = f.name
    try:
        write_safetensors(path, tensors)
        h = verify_safetensors(path, expected_names=[n for n, _ in tensors])
        assert '__metadata__' in h
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(0)
    print('Running CausalLM safetensors weight converter tests...')
    print()

    for fn in [
        test_header_is_8byte_aligned,
        test_metadata_format_field,
        test_roundtrip_values,
        test_multiple_dtypes,
        test_data_offsets_contiguous,
        test_qwen2_names,
        test_qwen3_names,
        test_gemma3_names,
        test_kalm_embedding_names,
        test_verify_safetensors,
    ]:
        run(fn)

    print()
    print(f'Results: {len(PASSED)} passed, {len(FAILED)} failed')
    if FAILED:
        print(f'FAILED: {FAILED}')
        sys.exit(1)
    else:
        print('ALL TESTS PASSED')
