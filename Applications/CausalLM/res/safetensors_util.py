# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics
#
# @file  safetensors_util.py
# @brief Python helper for nntrainer safetensors format
#        (shared by all CausalLM weight converters)
#
# Format on disk:
#   [uint64 LE header_size][JSON header padded to 8-byte boundary][tensor bytes...]
# Header JSON:
#   {"__metadata__":{"format":"nntrainer"},
#    "<nntrainer_name>":{"dtype":"F32","shape":[...],"data_offsets":[s,e]}, ...}
# nntrainer tensor name: "<layer_name>:<weight_role>"
#   e.g. "layer0_wq:weight", "layer0_attention_norm:gamma"

import json
import struct
import numpy as np

_NP_TO_NNTR = {
    'float32': 'F32',
    'float16': 'F16',
    'int8':    'I8',
    'int16':   'I16',
    'uint8':   'U8',
    'uint16':  'U16',
    'uint32':  'U32',
}


def write_safetensors(output_path, named_tensors):
    """Write list of (nntrainer_name, ndarray) to an nntrainer safetensors file."""
    offset = 0
    entries = []
    for name, arr in named_tensors:
        arr = np.ascontiguousarray(arr)
        dtype_code = _NP_TO_NNTR.get(str(arr.dtype), 'F32')
        nbytes = arr.nbytes
        entries.append((name, arr, dtype_code, offset, offset + nbytes))
        offset += nbytes

    header = {'__metadata__': {'format': 'nntrainer'}}
    for name, arr, dtype_code, off_s, off_e in entries:
        header[name] = {
            'dtype': dtype_code,
            'shape': list(arr.shape),
            'data_offsets': [off_s, off_e],
        }

    json_str = json.dumps(header, separators=(',', ':'))
    pad = (8 - (len(json_str) % 8)) % 8
    json_str += ' ' * pad
    json_bytes = json_str.encode('utf-8')

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<Q', len(json_bytes)))
        f.write(json_bytes)
        for _name, arr, _dt, _s, _e in entries:
            f.write(arr.tobytes())

    print(f'[safetensors] wrote {len(entries)} tensors -> {output_path}')


def verify_safetensors(path, expected_names=None):
    """Parse an nntrainer safetensors file and optionally check expected tensor names.
    Returns the parsed header dict.
    """
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size).decode('utf-8'))

    metadata = header.get('__metadata__', {})
    tensor_keys = [k for k in header if k != '__metadata__']

    print(f'[verify] {path}')
    print(f'  format  : {metadata.get("format", "?")}')  
    print(f'  tensors : {len(tensor_keys)}')

    if expected_names is not None:
        missing = set(expected_names) - set(tensor_keys)
        extra   = set(tensor_keys) - set(expected_names)
        if missing:
            print(f'  MISSING : {sorted(missing)}')
        if extra:
            print(f'  EXTRA   : {sorted(extra)}')
        if not missing and not extra:
            print(f'  OK - all {len(expected_names)} expected tensors present')

    show = sorted(tensor_keys)[:8]
    for key in show:
        info = header[key]
        size_kb = (info['data_offsets'][1] - info['data_offsets'][0]) / 1024
        print(f'  {key:<52s} {info["dtype"]} {info["shape"]} ({size_kb:.1f} KB)')
    if len(tensor_keys) > 8:
        print(f'  ... ({len(tensor_keys) - 8} more)')

    return header
