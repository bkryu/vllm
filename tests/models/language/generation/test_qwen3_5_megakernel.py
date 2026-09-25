# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit checks for the Qwen3.8 megakernel adapter (`--gdn-decode-backend
megakernel`): the NVFP4 block-scale unswizzle inverts vLLM's swizzle, the
backend knob resolves and validates, the per-head q|gate row permutation is a
permutation. The end-to-end path needs an SM120 GPU and the
nvidia/Qwen3.8-27B-NVFP4 checkpoint (see the module docstring of
vllm.models.qwen3_5.nvidia.megakernel)."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.models.qwen3_5.nvidia.megakernel import (
    GDN_DECODE_BACKENDS,
    resolve_gdn_decode_backend,
    unswizzle_blockscale,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="swizzle_blockscale moves to CUDA")
@pytest.mark.parametrize("n,k16", [(34816, 320), (5120, 1088), (200, 7)])
def test_unswizzle_blockscale_roundtrip(n, k16):
    scale = (torch.rand(n, k16, device="cuda") * 4).to(torch.float8_e4m3fn)
    back = unswizzle_blockscale(swizzle_blockscale(scale), n, k16)
    assert torch.equal(back.view(torch.uint8), scale.view(torch.uint8))


class _Cfg:
    def __init__(self, additional_config):
        self.additional_config = additional_config


def test_resolve_gdn_decode_backend():
    assert resolve_gdn_decode_backend(_Cfg({})) == "auto"
    assert resolve_gdn_decode_backend(_Cfg(None)) == "auto"
    for b in GDN_DECODE_BACKENDS:
        assert resolve_gdn_decode_backend(_Cfg({"gdn_decode_backend": b})) == b
    with pytest.raises(ValueError, match="Unknown gdn_decode_backend"):
        resolve_gdn_decode_backend(_Cfg({"gdn_decode_backend": "cutile"}))


def test_q_gate_row_permutation():
    hq, hkv, d = 24, 4, 256
    qd, kvd = hq * d, hkv * d
    dd = torch.arange(d)
    heads = torch.arange(hq)[:, None] * (2 * d)
    perm = torch.cat(
        [
            (heads + dd).reshape(-1),
            (heads + d + dd).reshape(-1),
            torch.arange(2 * qd, 2 * qd + 2 * kvd),
        ]
    )
    assert perm.shape[0] == 2 * qd + 2 * kvd
    assert torch.equal(perm.sort().values, torch.arange(2 * qd + 2 * kvd))
    # head h's q rows come first, then every head's gate rows, then k | v
    assert perm[0] == 0 and perm[d] == 2 * d and perm[qd] == d
