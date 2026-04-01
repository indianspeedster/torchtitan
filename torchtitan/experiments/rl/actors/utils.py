# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torchtitan.config import CompileConfig
from torchtitan.models.common.attention import VarlenMetadata


# TODO We should either unify all the mask creation for RL, or move them to a
#      single file.
def build_varlen_metadata(
    input_sequences: list[tuple[torch.Tensor, int, int]], device: torch.device
) -> VarlenMetadata:
    """Build VarlenMetadata for all sequences in a batch."""
    cu_seqs = torch.cumsum(
        torch.tensor(
            [0] + [token_ids.shape[0] for token_ids, _, _ in input_sequences],
            dtype=torch.int32,
            device=device,
        ),
        0,
        dtype=torch.int32,
    )
    max_len = max(token_ids.shape[0] for token_ids, _, _ in input_sequences)
    return VarlenMetadata(
        cu_seq_q=cu_seqs, cu_seq_k=cu_seqs, max_q=max_len, max_k=max_len
    )


def compute_token_log_probs(
    model: torch.nn.Module,
    prompt_ids: list[int],
    gen_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute per-token log probabilities for generated tokens.
    TODO Only batch size 1 is supported for now.

    Args:
        model: The model to use for computing logits
        prompt_ids: Prompt token IDs
        gen_ids: Generated token IDs
        device: Device to run computation on

    Returns:
        Per-token log probabilities for the generated tokens
    """
    token_ids = torch.tensor(prompt_ids + gen_ids, dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    gen_len = len(gen_ids)
    attention_masks = build_varlen_metadata([(token_ids, prompt_len, gen_len)], device)

    full_tensor = token_ids.unsqueeze(0)

    # NOTE: We should move towards batching to improve efficiency here
    # See https://github.com/pytorch/torchtitan/issues/2674
    # Explicit positions avoid dynamic rope_cache[0:seqlen] slice in RoPE,
    # which breaks torch.compile with symbolic shapes.
    seq_len = full_tensor.shape[1]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    logits = model(full_tensor, attention_masks=attention_masks, positions=positions)

    # Convert to float32 for numerical stability
    logits_f32 = logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits_f32, dim=-1)
    target_tokens = full_tensor[:, 1:]

    # Extract log probs for generated tokens only
    gen_start_idx = prompt_len - 1
    gen_end_idx = gen_start_idx + gen_len

    gen_token_logprobs = log_probs[0, gen_start_idx:gen_end_idx, :]
    gen_token_ids_tensor = target_tokens[0, gen_start_idx:gen_end_idx]
    token_lps = gen_token_logprobs.gather(
        1, gen_token_ids_tensor.unsqueeze(-1)
    ).squeeze(-1)

    return token_lps


def policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    kl_coef: float = 0.1,
    ppo_clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compile-friendly GRPO/PPO policy gradient loss on padded tensors.

    Operates on padded, batched log-prob tensors with a boolean mask for
    valid (non-padding) positions.  Returns scalar tensors — call ``.item()``
    outside the compiled region to avoid graph breaks.

    Args:
        policy_log_probs: [batch, max_gen_len] padded policy log probs
        ref_log_probs: [batch, max_gen_len] padded reference log probs (detached)
        mask: [batch, max_gen_len] True for valid token positions
        advantages: [batch] per-sample advantages
        kl_coef: KL divergence penalty coefficient
        ppo_clip_eps: PPO clipping epsilon
        entropy_coef: Entropy bonus coefficient

    Returns:
        (total_loss, pg_loss, entropy, kl_div, ratio_mean, ratio_clipped_frac)
    """
    # Per-token log ratio, zeroed at padding positions
    token_log_ratio = (policy_log_probs - ref_log_probs) * mask

    # Valid token count per sample (clamp avoids division by zero)
    token_counts = mask.sum(dim=1).clamp(min=1)  # [batch]

    # Per-sample mean log ratio
    mean_log_ratio = token_log_ratio.sum(dim=1) / token_counts  # [batch]

    # Per-token KL (Schulman approximation: ratio - 1 - log_ratio)
    token_ratio = torch.exp(token_log_ratio)
    token_kl = (token_ratio - 1 - token_log_ratio) * mask
    mean_kl = token_kl.sum(dim=1) / token_counts  # [batch]

    # PPO clipped objective
    ratio = torch.exp(mean_log_ratio)
    unclipped_loss = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps)
    clipped_loss = clipped_ratio * advantages
    pg_loss = -torch.min(unclipped_loss, clipped_loss).mean()

    # Entropy bonus (averaged across all valid tokens)
    total_valid = mask.sum().clamp(min=1)
    entropy = -(policy_log_probs * mask).sum() / total_valid
    entropy_bonus = -entropy_coef * entropy

    # KL divergence penalty (averaged across samples)
    kl_div = mean_kl.mean()

    # Total loss
    total_loss = pg_loss + entropy_bonus + kl_coef * kl_div

    # Metric tensors (no .item() — caller converts outside compiled region)
    ratio_mean = ratio.mean()
    ratio_clipped_frac = (torch.abs(ratio - clipped_ratio) > 1e-6).float().mean()

    return total_loss, pg_loss, entropy, kl_div, ratio_mean, ratio_clipped_frac


def build_policy_gradient_loss(
    compile_config: CompileConfig,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Optionally compile ``policy_gradient_loss`` following the build_*_loss pattern."""
    loss_fn = policy_gradient_loss
    if compile_config.enable and "loss" in compile_config.components:
        # if compile_config.enable and "loss" in compile_config.components:
        loss_fn = torch.compile(loss_fn, backend="inductor", fullgraph=True)
    return loss_fn


def compute_policy_gradient_loss(
    model: torch.nn.Module,
    vllm_token_ids: list[list[int]],
    prompt_token_ids: list[list[int]],
    advantages: torch.Tensor,
    ref_token_log_probs: list[torch.Tensor],
    loss_fn: Callable[..., tuple[torch.Tensor, ...]] | None = None,
    kl_coef: float = 0.1,
    ppo_clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, dict, list[torch.Tensor]]:
    """
    Compute GRPO/PPO policy gradient loss with per-token KL divergence.

    Uses per-token log ratios (averaged across tokens) instead of per-sequence
    sums to prevent ratio explosion when sequences are long.

    Args:
        model: Current policy model
        vllm_token_ids: Generated token IDs for each completion
        prompt_token_ids: Prompt token IDs for each completion
        advantages: [batch] - Advantages for each sample
        ref_token_log_probs: Per-token log probs from reference model (frozen)
        loss_fn: Compiled (or eager) loss function from build_policy_gradient_loss.
            Falls back to uncompiled policy_gradient_loss when None.
        kl_coef: KL divergence penalty coefficient
        ppo_clip_eps: PPO clipping epsilon
        entropy_coef: Entropy bonus coefficient

    Returns:
        loss: Total loss (PG + entropy + KL)
        metrics: Training metrics dict
        batch_token_log_probs: List of per-token log probs for each sample (for verification)
    """
    device = next(model.parameters()).device
    advantages = advantages.to(device)

    # Compute per-token log probs under current policy (WITH GRADIENTS)
    batch_token_log_probs = []

    for prompt_toks, gen_toks in zip(prompt_token_ids, vllm_token_ids):
        token_lps = compute_token_log_probs(
            model,
            prompt_toks,
            gen_toks,
            device,
        )
        batch_token_log_probs.append(token_lps)

    # Pad variable-length log probs into [batch, max_gen_len] tensors
    policy_log_probs_padded = pad_sequence(
        batch_token_log_probs, batch_first=True, padding_value=0.0
    )
    ref_log_probs_padded = pad_sequence(
        [r.detach() for r in ref_token_log_probs],
        batch_first=True,
        padding_value=0.0,
    )

    # Build mask: True for valid (non-padding) token positions
    lengths = torch.tensor([t.shape[0] for t in batch_token_log_probs], device=device)
    max_len = policy_log_probs_padded.shape[1]
    mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    # Call (optionally compiled) loss function
    if loss_fn is None:
        loss_fn = policy_gradient_loss

    total_loss, pg_loss, entropy, kl_div, ratio_mean, ratio_clipped_frac = loss_fn(
        policy_log_probs_padded,
        ref_log_probs_padded,
        mask,
        advantages,
        kl_coef,
        ppo_clip_eps,
        entropy_coef,
    )

    # Extract metrics outside compiled region
    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy.item(),
        "kl_div": kl_div.item(),
        "ratio_mean": ratio_mean.item(),
        "ratio_clipped_frac": ratio_clipped_frac.item(),
    }

    return total_loss, metrics, batch_token_log_probs


def verify_logprob_identity(
    vllm_token_log_probs: list[list[float]],
    batch_token_log_probs: list[torch.Tensor],
) -> dict:
    """
    Check if vLLM log probs and computed log probs are bit-wise identical,
    and compute the log ratio (train/generator) between them.

    Args:
        vllm_token_log_probs: Per-token log probs from vLLM (generator)
        batch_token_log_probs: Per-token log probs computed by the trainer model

    Returns:
        Verification result dict with identity status, delta info, and log ratio stats
    """
    result = {
        "logprob_bitwise_identical": True,
        "num_samples_checked": len(vllm_token_log_probs),
        "total_tokens_checked": 0,
        "num_tokens_different": 0,
        "logprob_max_delta": 0.0,
        "avg_delta": 0.0,
        "logprob_diff_mean": 0.0,
        "logprob_diff_max": 0.0,
    }

    all_deltas = []
    all_log_ratios = []

    for vllm_lps, titan_lps in zip(vllm_token_log_probs, batch_token_log_probs):
        # Convert vLLM log probs to tensor
        vllm_tensor = torch.tensor(vllm_lps, dtype=torch.float32)
        # Convert titan log probs to float32 for comparison
        titan_tensor = titan_lps.detach().cpu().float()

        num_tokens = len(vllm_lps)
        result["total_tokens_checked"] += num_tokens

        # Check bitwise identity
        bitwise_match = torch.equal(vllm_tensor, titan_tensor)

        if not bitwise_match:
            result["logprob_bitwise_identical"] = False
            num_different = (vllm_tensor != titan_tensor).sum().item()
            result["num_tokens_different"] += num_different
            deltas = (vllm_tensor - titan_tensor).abs()
            all_deltas.append(deltas)

        # Log ratio: log(pi_train / pi_generator) = logprob_train - logprob_generator
        # Should be 0 when weights are identical (ratio = 1)
        all_log_ratios.append(titan_tensor - vllm_tensor)

    # Compute aggregate delta stats
    if all_deltas:
        combined_deltas = torch.cat(all_deltas)
        result["logprob_max_delta"] = combined_deltas.max().item()
        result["avg_delta"] = combined_deltas.mean().item()

    # Compute log ratio stats
    if all_log_ratios:
        combined_log_ratios = torch.cat(all_log_ratios)
        result["logprob_diff_mean"] = combined_log_ratios.mean().item()
        result["logprob_diff_max"] = combined_log_ratios.abs().max().item()

    return result
