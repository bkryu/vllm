# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 decoder megakernel (FlashInfer experimental API) for Qwen3.5-family
hybrid decoders on SM120, TP=1.

A uniform decode step (every request contributes the same number of tokens:
1 for plain decode, K+1 for the MTP verify step) of the whole decoder — 48
gated-DeltaNet layers + 16 full-attention layers for
``nvidia/Qwen3.8-27B-NVFP4`` — runs as ONE persistent CuTe DSL launch
(``flashinfer.qwen3_ops.qwen38_megakernel_decode``) instead of the per-layer
kernel chain. Everything else (prefill, mixed batches, more than 16 rows, the
MTP drafter) stays on the stock path.

Opt in with ``--gdn-decode-backend megakernel``. The model-config hook forces
the paged-KV geometry the kernel needs (block size 32, FlashInfer attention
backend, text-only inputs); this module prepares the weights once (row
permutations, dtype views and unswizzled copies only — never a
re-quantization) and routes eligible steps.

State and cache tensors are consumed in vLLM's own layouts (SSM state
``[block, hv, dv, dk]``, conv state ``[block, rows, C]`` with the spec-decode
sliding window, K|V packed pages, per-token GDN state slots), so the stock
kernels and the megakernel can alternate step by step.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

GDN_DECODE_BACKENDS = ("auto", "megakernel", "cuda", "triton")
MAX_ROWS = 16  # decode rows per launch the kernel serves
DEBUG = os.environ.get("VLLM_QWEN38_MEGAKERNEL_DEBUG", "0") == "1"
MAX_SPEC_ROWS = 4  # rows per sequence in the verify form (num_speculative_tokens + 1)


def resolve_gdn_decode_backend(vllm_config: VllmConfig) -> str:
    """``--gdn-decode-backend`` (via ``additional_config``), validated."""
    additional = vllm_config.additional_config
    backend = (
        additional.get("gdn_decode_backend", "auto")
        if isinstance(additional, dict)
        else "auto"
    )
    if backend not in GDN_DECODE_BACKENDS:
        raise ValueError(
            f"Unknown gdn_decode_backend {backend!r}; expected one of "
            f"{GDN_DECODE_BACKENDS}"
        )
    return backend


def megakernel_requested(vllm_config: VllmConfig) -> bool:
    return resolve_gdn_decode_backend(vllm_config) == "megakernel"


def megakernel_unsupported_reason(vllm_config: VllmConfig) -> str | None:
    """Why this deployment cannot use the megakernel (None = it can). Config-level
    checks only; per-layer weight formats are validated when the weights are
    prepared."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return "CUDA only"
    if not current_platform.is_device_capability_family(120):
        return "SM120 (RTX PRO 6000 / GeForce Blackwell) only"
    pc = vllm_config.parallel_config
    if pc.tensor_parallel_size != 1 or pc.pipeline_parallel_size != 1:
        return "TP=1 / PP=1 only"
    hf = vllm_config.model_config.hf_text_config
    if getattr(hf, "model_type", "") not in ("qwen3_5", "qwen3_5_text"):
        return f"Qwen3.5-family dense text decoder only (model_type {hf.model_type!r})"
    if (
        getattr(hf, "linear_key_head_dim", 0) != 128
        or getattr(hf, "linear_value_head_dim", 0) != 128
        or getattr(hf, "linear_conv_kernel_dim", 0) != 4
        or getattr(hf, "head_dim", 0) != 256
    ):
        return "GDN head dims 128 / conv width 4 / attention head_dim 256 only"
    if hf.hidden_size % 256 or hf.intermediate_size % 64:
        return "hidden_size % 256 == 0 and intermediate_size % 64 == 0 required"
    if not getattr(hf, "attn_output_gate", True):
        return "attention output gate expected"
    sc = vllm_config.speculative_config
    if sc is not None and sc.num_speculative_tokens + 1 > MAX_SPEC_ROWS:
        return f"num_speculative_tokens <= {MAX_SPEC_ROWS - 1}"
    try:
        import flashinfer.qwen3_ops  # noqa: F401
    except ImportError as e:
        return f"flashinfer.qwen3_ops is not importable ({e})"
    return None


def text_only_multimodal_reason(vllm_config: VllmConfig) -> str | None:
    """The kernel rotates with 1-D positions: multimodal (mRoPE) requests must be
    disabled so every request is text-only."""
    mc = vllm_config.model_config
    if not mc.is_multimodal_model:
        return None
    mm = mc.multimodal_config
    limits = getattr(mm, "limit_per_prompt", None) or {}

    def _count(v):
        if isinstance(v, dict):
            return v.get("count", 0)
        return getattr(v, "count", v)

    if limits and all(_count(v) == 0 for v in limits.values()):
        return None
    return (
        "the megakernel serves text-only requests (1-D rotary positions); "
        'pass --limit-mm-per-prompt \'{"image": 0, "video": 0}\''
    )


# ------------------------------------------------------------- weight preparation


def _fp8_nk(linear, name: str) -> torch.Tensor:
    """ModelOpt per-tensor fp8 linear after loading stores ``weight`` as the [K, N]
    transpose view of its [N, K] storage; give the kernel the [N, K] view."""
    w = linear.weight
    if w.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{name}: expected per-tensor fp8 weights, got {w.dtype}")
    w = w.t() if w.stride(0) == 1 else w
    if not w.is_contiguous():
        w = w.contiguous()
    return w


def _fp8_scale(linear, name: str) -> float:
    s = linear.weight_scale
    if s.numel() != 1:
        raise ValueError(
            f"{name}: expected one per-tensor fp8 scale, got {tuple(s.shape)}"
        )
    return float(s.float().reshape(-1)[0])


def unswizzle_blockscale(swizzled: torch.Tensor, n: int, k16: int) -> torch.Tensor:
    """Inverse of ``swizzle_blockscale`` (CUTLASS 128x4 block interleave) -> plain
    [n, k16] e4m3 scales."""
    m_pad, k_pad = swizzled.shape
    if m_pad % 128 or k_pad % 4 or m_pad < n or k_pad < k16:
        raise ValueError(
            f"unexpected NVFP4 block-scale shape {tuple(swizzled.shape)} "
            f"for [{n}, {k16}]"
        )
    plain = (
        swizzled.reshape(1, m_pad // 128, k_pad // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(m_pad, k_pad)
    )
    return plain[:n, :k16].contiguous()


def _nvfp4_parts(linear, name: str):
    """(packed weight uint8 [N, K/2], plain block scales [N, K/16], global scale) of a
    ModelOpt NVFP4 linear processed by the CUTLASS/FlashInfer kernel path."""
    w = linear.weight
    if w.dtype != torch.uint8 or w.dim() != 2:
        raise ValueError(
            f"{name}: expected NVFP4 packed uint8 weights, got {w.dtype} "
            f"{tuple(w.shape)}"
        )
    n, kh = w.shape
    scale = linear.weight_scale
    if scale.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{name}: expected e4m3 NVFP4 block scales, got {scale.dtype}")
    gs = getattr(linear, "weight_global_scale", None)
    if gs is None:
        raise ValueError(
            f"{name}: missing weight_global_scale (NVFP4 weights not post-processed?)"
        )
    k16 = 2 * kh // 16
    if tuple(scale.shape) == (n, k16) and n % 128:
        plain = scale.contiguous()  # already plain (kernel paths that do not swizzle)
    else:
        plain = unswizzle_blockscale(scale, n, k16)
    return w.contiguous(), plain, float(gs.float().reshape(-1)[0])


def _plus_one(norm) -> torch.Tensor:
    """Qwen3.5's zero-centered RMSNorm weights (y = norm(x) * (1 + w)), folded for
    the kernel."""
    return (norm.weight.float() + 1.0).to(torch.bfloat16).contiguous()


class _NotReady(RuntimeError):
    pass


class Qwen38MegakernelRunner:
    """Owns the prepared per-layer tensors, the plans per (rows, spec rows) and the
    per-step metadata glue for one ``Qwen3_5Model``."""

    def __init__(self, model, vllm_config: VllmConfig) -> None:
        self.model = model
        self.vllm_config = vllm_config
        hf = vllm_config.model_config.hf_text_config
        self.H = hf.hidden_size
        self.I = hf.intermediate_size
        self.hk, self.hv = hf.linear_num_key_heads, hf.linear_num_value_heads
        self.hq, self.hkv, self.d = (
            hf.num_attention_heads,
            hf.num_key_value_heads,
            hf.head_dim,
        )
        rp = getattr(hf, "rope_parameters", None) or {}
        self.rot = int(
            getattr(hf, "partial_rotary_factor", rp.get("partial_rotary_factor", 1.0))
            * self.d
        )
        self.theta = float(rp.get("rope_theta", getattr(hf, "rope_theta", 10000.0)))
        self.eps = float(hf.rms_norm_eps)
        # kernel block (page) size the kernel serves; the config hook forces it
        self.page = 32
        self.max_model_len = vllm_config.model_config.max_model_len
        sc = vllm_config.speculative_config
        self.max_R = (sc.num_speculative_tokens + 1) if sc is not None else 1
        self.layers: list[dict[str, Any]] | None = None
        self.plans: dict[tuple[int, int], Any] = {}
        self.specs: dict[int, Any] = {}
        self.slot_bufs: dict[tuple[int, int], torch.Tensor] = {}
        self.acc_bufs: dict[int, torch.Tensor] = {}
        self.ones: torch.Tensor | None = None
        self.gdn_groups: list[list[int]] | None = (
            None  # GDN layer indices per KV-cache group
        )
        self.attn_name: str | None = None
        self.disabled_reason: str | None = None
        self._logged: set[str] = set()

    # ------------------------------------------------------------------ preparation
    def _log_once(self, key: str, msg: str, *args) -> None:
        if key not in self._logged:
            self._logged.add(key)
            logger.info(msg, *args)

    def _prepare(self) -> None:
        from flashinfer.experimental.qwen38_megakernel import (
            build_rope_cache,
            interleave_gate_up,
        )

        model = self.model
        dev = model.embed_tokens.weight.device
        layers = []
        n = len(model.layers)
        for layer in model.layers:  # KV caches are bound after the profile run
            if layer.layer_type == "linear_attention":
                if not isinstance(getattr(layer.linear_attn, "kv_cache", None), tuple):
                    raise _NotReady("GDN state pools not bound yet")
            elif layer.self_attn.attn.kv_cache.numel() == 0:
                raise _NotReady("attention KV cache not bound yet")
        for j, layer in enumerate(model.layers):
            nxt = model.layers[j + 1].input_layernorm if j + 1 < n else model.norm
            m = layer.mlp
            w_gu, s_gu, a_gu = _nvfp4_parts(m.gate_up_proj, f"layer {j} gate_up_proj")
            w_il, s_il = interleave_gate_up(w_gu, s_gu, self.I)
            w_dn, s_dn, a_dn = _nvfp4_parts(m.down_proj, f"layer {j} down_proj")
            ly: dict[str, Any] = dict(
                w_gu_il=w_il,
                s_gu_il=s_il,
                alpha_gu=torch.tensor([a_gu, a_gu], dtype=torch.float32, device=dev),
                w_dn=w_dn,
                s_dn=s_dn,
                alpha_dn=torch.tensor([a_dn], dtype=torch.float32, device=dev),
                wn_mid=_plus_one(layer.post_attention_layernorm),
                wn_out=_plus_one(nxt),
            )
            if layer.layer_type == "linear_attention":
                g = layer.linear_attn
                s_in = _fp8_scale(g.in_proj_qkvz, f"layer {j} in_proj_qkvz")
                if g.in_proj_ba.weight.dtype != torch.bfloat16:
                    raise ValueError(f"layer {j} in_proj_ba: expected bf16 weights")
                conv_pool, ssm_pool = g.kv_cache[0], g.kv_cache[1]
                ly.update(
                    kind=0,
                    w_in=_fp8_nk(g.in_proj_qkvz, f"layer {j} in_proj_qkvz"),
                    alpha_in=torch.tensor(
                        [s_in, s_in], dtype=torch.float32, device=dev
                    ),
                    w_out=_fp8_nk(g.out_proj, f"layer {j} out_proj"),
                    alpha_out=torch.tensor(
                        [_fp8_scale(g.out_proj, f"layer {j} out_proj")],
                        dtype=torch.float32,
                        device=dev,
                    ),
                    w_ba=g.in_proj_ba.weight.contiguous(),
                    conv_w=g.conv1d.weight.reshape(
                        g.conv_dim, g.conv_kernel_size
                    ).contiguous(),
                    a_log=g.A_log.float().contiguous(),
                    dt_bias=g.dt_bias.float().contiguous(),
                    norm_w=g.norm.weight.to(torch.bfloat16).contiguous(),
                    conv_pool=conv_pool,
                    ssm_pool=ssm_pool,
                    # assigned from the KV-cache groups at the first eligible step
                    slot_group=0,
                    _name=g.prefix,
                )
            else:
                a = layer.self_attn
                qd, kvd = self.hq * self.d, self.hkv * self.d
                w = _fp8_nk(a.qkv_proj, f"layer {j} qkv_proj")
                if w.shape[0] != 2 * qd + 2 * kvd:
                    raise ValueError(
                        f"layer {j} qkv_proj rows {w.shape[0]} != 2*q + k + v "
                        "with the output gate"
                    )
                # vLLM packs [q_h | gate_h] per head; the kernel wants q | gate | k | v
                # (a row permutation)
                d = torch.arange(self.d, device=dev)
                heads = torch.arange(self.hq, device=dev)[:, None] * (2 * self.d)
                perm = torch.cat(
                    [
                        (heads + d).reshape(-1),
                        (heads + self.d + d).reshape(-1),
                        torch.arange(2 * qd, 2 * qd + 2 * kvd, device=dev),
                    ]
                )
                s_in = _fp8_scale(a.qkv_proj, f"layer {j} qkv_proj")
                kvp = self._attn_pages(a, j)
                ly.update(
                    kind=1,
                    w_in=w[perm].contiguous(),
                    alpha_in=torch.tensor([s_in] * 4, dtype=torch.float32, device=dev),
                    w_out=_fp8_nk(a.o_proj, f"layer {j} o_proj"),
                    alpha_out=torch.tensor(
                        [_fp8_scale(a.o_proj, f"layer {j} o_proj")],
                        dtype=torch.float32,
                        device=dev,
                    ),
                    wq=_plus_one(a.q_norm),
                    wk=_plus_one(a.k_norm),
                    k_cache=kvp[..., : self.d],
                    v_cache=kvp[..., self.d :],
                    _name=a.attn.layer_name,
                )
                if self.attn_name is None:
                    self.attn_name = a.attn.layer_name
            layers.append(ly)
        self.layers = layers
        self.cache_sig = self._cache_signature()
        self.wn_in = _plus_one(model.layers[0].input_layernorm)
        self.wn_out = _plus_one(model.norm)
        self.rope_table = build_rope_cache(
            self.max_model_len + 64 * self.max_R, self.rot, theta=self.theta, device=dev
        )
        self.ones = torch.ones(MAX_ROWS, dtype=torch.int32, device=dev)
        extra = sum(
            t.numel() * t.element_size()
            for ly in layers
            for k, t in ly.items()
            if k in ("w_gu_il", "s_gu_il", "s_dn") or (k == "w_in" and ly["kind"] == 1)
        )
        logger.info(
            "Qwen3.8 megakernel: prepared %d layers (%.2f GB of permuted / "
            "unswizzled weight copies); "
            "rotary dim %d theta %.0f, page %d, max spec rows %d",
            len(layers),
            extra / 1e9,
            self.rot,
            self.theta,
            self.page,
            self.max_R,
        )

    def _attn_pages(self, a, j: int) -> torch.Tensor:
        """The attention layer's KV cache as [blocks, tokens, heads, 2D]
        (K | V packed)."""
        kv = a.attn.kv_cache
        if (
            kv.dim() != 4
            or kv.shape[1] != self.hkv
            or kv.shape[2] != self.page
            or kv.shape[3] != 2 * self.d
        ):
            raise ValueError(
                f"layer {j}: attention KV cache {tuple(kv.shape)} is not "
                f"[blocks, {self.hkv}, {self.page}, {2 * self.d}] "
                "(FlashInfer backend, block size 32 required)"
            )
        kvp = kv.permute(0, 2, 1, 3)  # contiguous in the NHD physical layout
        if not kvp.is_contiguous():
            raise ValueError(
                f"layer {j}: KV cache layout is not NHD "
                "(unset VLLM_KV_CACHE_LAYOUT or use NHD)"
            )
        return kvp

    def _cache_signature(self):
        """Identity of the bound caches: vLLM binds a small KV cache for the
        compile / profile passes and the real one afterwards."""
        sig = []
        for layer in self.model.layers:
            if layer.layer_type == "linear_attention":
                kc = layer.linear_attn.kv_cache
                sig.append((kc[1].data_ptr(), tuple(kc[1].shape)))
            else:
                kv = layer.self_attn.attn.kv_cache
                sig.append((kv.data_ptr(), tuple(kv.shape)))
        return tuple(sig)

    def _refresh_caches(self) -> None:
        """Re-point the prepared layers at the currently bound caches (weights kept)."""
        for j, (layer, ly) in enumerate(zip(self.model.layers, self.layers)):
            if ly["kind"] == 0:
                g = layer.linear_attn
                ly["conv_pool"], ly["ssm_pool"] = g.kv_cache[0], g.kv_cache[1]
            else:
                kvp = self._attn_pages(layer.self_attn, j)
                ly["k_cache"], ly["v_cache"] = kvp[..., : self.d], kvp[..., self.d :]
        self.plans.clear()  # the device tables hold cache pointers
        self.cache_sig = self._cache_signature()
        kinds = [ly["kind"] for ly in self.layers]
        self._log_once(
            f"caches-{self.cache_sig[0][0]}",
            "Qwen3.8 megakernel: caches re-bound (%d attention pages, "
            "%d GDN state blocks)",
            self.layers[kinds.index(1)]["k_cache"].shape[0],
            self.layers[kinds.index(0)]["ssm_pool"].shape[0],
        )

    def _spec(self, R: int):
        spec = self.specs.get(R)
        if spec is None:
            from flashinfer.experimental.qwen38_megakernel import DecoderSpec

            spec = self.specs[R] = DecoderSpec(
                self.hk,
                self.hv,
                self.hq,
                self.hkv,
                self.d,
                self.rot,
                self.page,
                self.H,
                self.I,
                splits=2,
                eps=self.eps,
                spec_rows=R,
                state_vk=True,
                conv_sd=True,
                kv_packed=True,
                slot_table=True,
            )
        return spec

    def _assign_groups(self, attn_md: dict) -> bool:
        """GDN layers sharing a metadata object share a KV-cache group (block table)."""
        groups: dict[int, list[int]] = {}
        for j, ly in enumerate(self.layers):
            if ly["kind"] != 0:
                continue
            md = attn_md.get(ly["_name"])
            if md is None:
                return False
            groups.setdefault(id(md), []).append(j)
        self.gdn_groups = list(groups.values())
        for gi, members in enumerate(self.gdn_groups):
            for j in members:
                self.layers[j]["slot_group"] = gi
        self.plans.clear()  # tables carry slot_group
        return True

    # ------------------------------------------------------------------ per step
    def maybe_forward(
        self, input_ids, positions, inputs_embeds=None
    ) -> torch.Tensor | None:
        """The normed hidden states of a uniform decode step, or None when the step is
        not eligible (prefill / mixed batch / too many rows / metadata missing)."""
        if self.disabled_reason is not None:
            return None
        ctx = get_forward_context()
        attn_md = ctx.attn_metadata
        if not isinstance(attn_md, dict):
            return None
        M = int(positions.shape[-1])
        if M > MAX_ROWS:
            return None
        if self.layers is None:
            try:
                self._prepare()
            except (
                AttributeError,
                IndexError,
                TypeError,
                RuntimeError,
            ) as e:  # KV caches not bound yet (profile run)
                self._log_once(
                    "prep-defer", "Qwen3.8 megakernel: preparation deferred (%s)", e
                )
                return None
            except ValueError as e:
                self.disabled_reason = str(e)
                logger.warning("Qwen3.8 megakernel DISABLED: %s", e)
                return None
        mda = attn_md.get(self.attn_name)
        if (
            mda is None
            or getattr(mda, "num_prefills", 1) != 0
            or getattr(mda, "num_decode_tokens", -1) <= 0
        ):
            return None
        if self._cache_signature() != self.cache_sig:
            try:
                self._refresh_caches()
            except ValueError as e:
                self.disabled_reason = str(e)
                logger.warning("Qwen3.8 megakernel DISABLED: %s", e)
                return None
        if self.gdn_groups is None and not self._assign_groups(attn_md):
            return None
        bt = getattr(mda, "block_table_tensor", None)
        seq_lens = getattr(mda, "seq_lens", None)
        if bt is None or seq_lens is None:
            self._log_once(
                "no-bt",
                "Qwen3.8 megakernel: FlashInfer metadata carries no block table; "
                "stock path",
            )
            return None
        # rows per sequence from the GDN metadata of the first group (spec verify:
        # K+1 rows per request)
        gmd = attn_md.get(self.layers[self.gdn_groups[0][0]]["_name"])
        if gmd is None or gmd.num_prefills != 0:
            return None
        if gmd.num_spec_decodes > 0:
            if (
                gmd.num_decodes != 0
                or gmd.spec_state_indices_tensor is None
                or gmd.num_accepted_tokens is None
            ):
                return None
            R = int(gmd.spec_state_indices_tensor.shape[1])
            if R > MAX_SPEC_ROWS or M % R:
                return None
        else:
            R = 1
        nseq = M // R
        if bt.shape[0] < nseq or seq_lens.shape[0] < nseq:
            return None
        if mda.slot_mapping.shape[0] < M:
            return None
        slots, acc = self._slot_table(attn_md, M, R, nseq)
        if slots is None:
            return None
        plan = self.plans.get((M, R))
        if plan is None:
            from flashinfer.qwen3_ops import qwen38_megakernel_prepare

            plan = self.plans[(M, R)] = qwen38_megakernel_prepare(
                self._spec(R),
                self.layers,
                M,
                max_seq_len=self.max_model_len,
                wn_in=self.wn_in,
                wn_out=self.wn_out,
                rope_table=self.rope_table,
            )
            self._log_once(
                f"plan-{M}-{R}",
                "Qwen3.8 megakernel: plan for %d rows x spec rows %d (kv splits %d)",
                M,
                R,
                plan.kv_splits,
            )
        from flashinfer.qwen3_ops import qwen38_megakernel_decode

        if DEBUG and not torch.cuda.is_current_stream_capturing():
            self._debug_check(
                M, R, nseq, bt, seq_lens, mda.slot_mapping, slots, acc, positions
            )
        hidden = (
            inputs_embeds
            if inputs_embeds is not None
            else self.model.embed_input_ids(input_ids)
        )
        pos = (
            positions[0] if positions.dim() == 2 else positions
        )  # mRoPE (3, T): text-only rows are identical
        return qwen38_megakernel_decode(
            plan,
            hidden.contiguous(),
            pos[:M].to(torch.int32),
            mda.slot_mapping[:M].to(torch.int32),
            seq_lens[:nseq].to(torch.int32),
            bt[:nseq].to(torch.int32),
            slots,
            num_accepted=acc,
        )

    def _debug_check(
        self, M, R, nseq, bt, seq_lens, slot_mapping, slots, acc, positions
    ):
        """VLLM_QWEN38_MEGAKERNEL_DEBUG=1: range-check every step (device syncs;
        eager only)."""
        first_attn = next(ly for ly in self.layers if ly["kind"] == 1)
        first_gdn = next(ly for ly in self.layers if ly["kind"] == 0)
        n_pages = first_attn["k_cache"].shape[0]
        n_slots = first_gdn["ssm_pool"].shape[0]
        sl = seq_lens[:nseq]
        pos = positions[0] if positions.dim() == 2 else positions
        stats = dict(
            seq_lens=(int(sl.min()), int(sl.max())),
            positions=(int(pos[:M].min()), int(pos[:M].max())),
            block_ids=(int(bt[:nseq].min()), int(bt[:nseq].max())),
            slot_mapping=(int(slot_mapping[:M].min()), int(slot_mapping[:M].max())),
            state_slots=(int(slots.min()), int(slots.max())),
            acc=(int(acc.min()), int(acc.max())),
            bt_width=int(bt.shape[1]),
            n_pages=n_pages,
            n_slots=n_slots,
            rope_rows=int(self.rope_table.shape[0]),
        )
        logger.info("Qwen3.8 megakernel DEBUG step M=%d R=%d: %s", M, R, stats)
        bad = []
        if stats["positions"][1] >= stats["rope_rows"]:
            bad.append("positions beyond the rope table")
        if stats["block_ids"][1] >= n_pages or stats["block_ids"][0] < 0:
            bad.append("block id outside the KV cache")
        if stats["slot_mapping"][1] >= n_pages * self.page:
            bad.append("slot mapping outside the KV cache")
        if stats["state_slots"][1] >= n_slots or stats["state_slots"][0] < 0:
            bad.append("state slot outside the pools")
        if stats["seq_lens"][1] > stats["bt_width"] * self.page:
            bad.append("seq_len beyond the block-table width")
        if bad:
            raise RuntimeError(f"Qwen3.8 megakernel DEBUG: {bad} ({stats})")

    def _slot_table(self, attn_md: dict, M: int, R: int, nseq: int):
        """[groups, nseq, R] int32 per-token GDN state slots (+ accepted counts) from
        the groups' metadata, in persistent buffers (CUDA-graph safe copies)."""
        G = len(self.gdn_groups)
        dev = self.rope_table.device
        buf = self.slot_bufs.get((M, R))
        if buf is None:
            buf = self.slot_bufs[(M, R)] = torch.zeros(
                G, nseq, R, dtype=torch.int32, device=dev
            )
        for gi, members in enumerate(self.gdn_groups):
            md = attn_md.get(self.layers[members[0]]["_name"])
            if md is None:
                return None, None
            if R > 1:
                idx = md.spec_state_indices_tensor
                if idx is None or idx.shape[0] < nseq or idx.shape[1] < R:
                    return None, None
                buf[gi].copy_(idx[:nseq, :R])
            else:
                idx = md.non_spec_state_indices_tensor
                if idx is None or idx.shape[0] < nseq:
                    return None, None
                buf[gi, :, 0].copy_(idx[:nseq])
        if R > 1:
            acc = self.acc_bufs.get(nseq)
            if acc is None:
                acc = self.acc_bufs[nseq] = torch.ones(
                    nseq, dtype=torch.int32, device=dev
                )
            src = attn_md.get(
                self.layers[self.gdn_groups[0][0]]["_name"]
            ).num_accepted_tokens
            if src is None or src.shape[0] < nseq:
                return None, None
            acc.copy_(src[:nseq])
        else:
            acc = self.ones[:nseq]
        return buf, acc
