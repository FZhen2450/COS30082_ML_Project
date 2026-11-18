"""Versioned, serialization-independent identities for dense model state tensors."""
import hashlib
import json
import struct
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

from final_protocol import file_sha256

STATE_NAMESPACE = 'cos30082-state-dict-sha256-v1'
STATE_DEFINITION = (
    'SHA256(namespace UTF8 plus NUL; tensors in ascending key order; for each tensor: '
    'uint64 big-endian metadata-byte length, compact sorted UTF8 JSON '
    '{name,dtype,shape,byte_order:little}, uint64 big-endian data-byte length, '
    'detached dense contiguous CPU tensor bytes in C order and little endian). '
    'Includes parameters and persistent buffers; excludes checkpoint metadata and serialization.'
)


def state_dict_identity(state):
    if sys.byteorder != 'little':
        raise ValueError('This identity version requires a little-endian host')
    if not isinstance(state, Mapping) or not state or any(not isinstance(key, str) for key in state):
        raise ValueError('Expected a nonempty named tensor state dictionary')
    digest = hashlib.sha256((STATE_NAMESPACE+'\0').encode('utf-8'))
    manifest = []
    elements = 0
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError('Only dense, non-quantized tensor state entries are supported')
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('State contains non-finite tensor values')
        metadata = {'name':name,'dtype':str(value.dtype),'shape':list(value.shape),'byte_order':'little'}
        header = json.dumps(metadata,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order='C')
        digest.update(struct.pack('>Q',len(header)))
        digest.update(header)
        digest.update(struct.pack('>Q',len(raw)))
        digest.update(raw)
        manifest.append({**metadata,'tensor_bytes_sha256':hashlib.sha256(raw).hexdigest()})
        elements += value.numel()
    return {'namespace':STATE_NAMESPACE,'definition':STATE_DEFINITION,
            'state_dict_sha256':digest.hexdigest(),'tensor_count':len(manifest),
            'state_elements':elements,'tensors':manifest}


def checkpoint_identity(path, expected_sha256=None):
    path = Path(path)
    before = file_sha256(path)
    if expected_sha256 is not None and before != expected_sha256:
        raise ValueError('Checkpoint file differs from the frozen release')
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or 'state_dict' not in payload:
        raise ValueError('Expected a state_dict checkpoint envelope')
    identity = state_dict_identity(payload['state_dict'])
    if file_sha256(path) != before:
        raise ValueError('Checkpoint file changed while computing its state identity')
    return {'checkpoint_file_sha256':before,**identity}
