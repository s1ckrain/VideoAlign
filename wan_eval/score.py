"""
Phase B: score generated videos with VideoReward (KwaiVGI/VideoReward).

Reward computation reproduces the standard call used by DanceGRPO / SAGE-GRPO:

    reward = inferencer.reward([video_path], [prompt], use_norm=True)
    composite = vq_coef * reward['VQ'] + mq_coef * reward['MQ'] + ta_coef * reward['TA']
    # default coefs (1.0, 1.0, 1.0) == SAGE-GRPO codebase default
    # (see SAGE-GRPO/hyvideo/models/reward_models/rewards.py:383)
    # any rewarding failure -> sentinel value -1.0 (DanceGRPO's fallback)

Weight references:
    DanceGRPO official          : vq=1.0, mq=0.0, ta=0.0   (VQ only -- conservative)
    SAGE-GRPO codebase default  : vq=1.0, mq=1.0, ta=1.0   <-- our default
    SAGE-GRPO paper Setting B   : vq=0.5, mq=0.5, ta=1.0   (special-tuned for HunyuanVideo)

Input : {output_dir}/videos_meta.csv     (from Phase A)
Output: {output_dir}/reward_scores.csv
        = videos_meta + reward_{VQ, MQ, TA, Overall, composite}
        where `reward_composite` is the EXACT training signal that GRPO would
        feed into its advantage computation under the chosen weights.

Usage (run from inside VideoAlign/wan_eval/):
    # Default: 1:1:1 averaging (SAGE-GRPO codebase default)
    python score.py --videos_meta outputs/videos_meta.csv --gpu 0

    # DanceGRPO compatibility (VQ only):
    python score.py --videos_meta ... --vq_coef 1.0 --mq_coef 0.0 --ta_coef 0.0 --gpu 0

    # --videoalign_dir defaults to ".." (i.e. the VideoAlign repo root).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score Wan-generated videos with VideoReward.")
    p.add_argument("--videos_meta", required=True,
                   help="CSV produced by Phase A (must contain `video_path`, `prompt`).")
    p.add_argument("--videoalign_dir", default="..",
                   help="Path to KwaiVGI VideoAlign repo root (provides `inference.py`). "
                        "Defaults to '..' since this script lives at VideoAlign/wan_eval/.")
    p.add_argument("--checkpoints_dir", default=None,
                   help="VideoReward checkpoint root. Defaults to {videoalign_dir}/checkpoints.")
    p.add_argument("--output_csv", default=None,
                   help="Where to save scores. Defaults to {videos_meta dir}/reward_scores.csv.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Score only the first N rows.")
    # Composite weights for the GRPO training signal.
    # Default = SAGE-GRPO codebase default (1.0, 1.0, 1.0), i.e. equal averaging.
    p.add_argument("--vq_coef", type=float, default=1.0,
                   help="Weight on VQ. Default 1.0 (SAGE-GRPO codebase default).")
    p.add_argument("--mq_coef", type=float, default=1.0,
                   help="Weight on MQ. Default 1.0 (SAGE-GRPO codebase default). "
                        "DanceGRPO official uses 0.0.")
    p.add_argument("--ta_coef", type=float, default=1.0,
                   help="Weight on TA. Default 1.0 (SAGE-GRPO codebase default). "
                        "DanceGRPO official uses 0.0 (TA is never read).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"--gpu={args.gpu} out of range (have {torch.cuda.device_count()} devices)")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    # Make VideoAlign importable.
    videoalign_dir = Path(args.videoalign_dir).resolve()
    if not (videoalign_dir / "inference.py").exists():
        raise FileNotFoundError(
            f"{videoalign_dir / 'inference.py'} not found. "
            "Pass the correct --videoalign_dir."
        )
    # Inserting BOTH the parent dir (so `from inference import ...` works) and
    # the dir itself (so the inference.py's own `from utils import ...` resolves).
    sys.path.insert(0, str(videoalign_dir))
    from inference import VideoVLMRewardInference  # noqa: E402

    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else videoalign_dir / "checkpoints"
    if not (checkpoints_dir / "model_config.json").exists():
        raise FileNotFoundError(
            f"{checkpoints_dir / 'model_config.json'} not found. "
            "Did you download KwaiVGI/VideoReward and place it under checkpoints/?"
        )

    # Load metadata.
    meta_csv = Path(args.videos_meta).resolve()
    meta_parent = meta_csv.parent
    df = pd.read_csv(meta_csv)
    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
    for col in ("video_path", "prompt"):
        if col not in df.columns:
            raise ValueError(f"`{col}` column missing from {meta_csv}")

    print(f"[score] loading VideoReward from {checkpoints_dir} ...")
    inferencer = VideoVLMRewardInference(
        str(checkpoints_dir), device=device, dtype=torch.bfloat16,
    )

    # Add output columns up-front so we can write checkpoints incrementally.
    for col in ("reward_VQ", "reward_MQ", "reward_TA", "reward_Overall", "reward_composite"):
        if col not in df.columns:
            df[col] = float("nan")

    output_csv = Path(args.output_csv) if args.output_csv else meta_parent / "reward_scores.csv"

    # Persist composite weights in metadata for downstream `accuracy.py`.
    coefs_path = output_csv.with_suffix(".coefs.json")
    import json as _json
    coefs = {"vq_coef": args.vq_coef, "mq_coef": args.mq_coef, "ta_coef": args.ta_coef}
    coefs_path.write_text(_json.dumps(coefs, indent=2))

    _presets = {
        (1.0, 0.0, 0.0): "DanceGRPO official (VQ only)",
        (1.0, 1.0, 1.0): "SAGE-GRPO codebase default (1:1:1 averaging)",
        (0.5, 0.5, 1.0): "SAGE-GRPO paper Setting B (HunyuanVideo-tuned)",
    }
    _key = (args.vq_coef, args.mq_coef, args.ta_coef)
    _label = _presets.get(_key, "custom")
    print(f"[score] composite coefs: {coefs}  -- {_label}")

    # Mini-batch loop (same pattern as VideoAlign's eval_videogen_rewardbench.py).
    # DanceGRPO calls `reward()` one video at a time; we batch for GPU efficiency
    # but the per-video output is identical (no cross-video coupling inside VideoReward).
    batch_paths: list[str] = []
    batch_prompts: list[str] = []
    batch_indices: list[int] = []

    pbar = tqdm(df.iterrows(), total=len(df), desc="Scoring")
    for idx, row in pbar:
        # `video_path` is stored relative to the metadata dir.
        full_path = str((meta_parent / row["video_path"]).resolve())
        batch_paths.append(full_path)
        batch_prompts.append(row["prompt"])
        batch_indices.append(idx)

        flush = (len(batch_paths) == args.batch_size) or (idx == len(df) - 1)
        if not flush:
            continue
        # DanceGRPO wraps `.reward()` in try/except and falls back to -1.0 on
        # failure (e.g. corrupt/missing video). Mirror that behaviour so the
        # scores we evaluate are *bit-identical* to what training would see.
        try:
            with torch.no_grad():
                # `use_norm=True` is the default of VideoVLMRewardInference.reward
                # -- same as DanceGRPO's explicit call.
                rewards = inferencer.reward(batch_paths, batch_prompts)
        except Exception as e:  # noqa: BLE001
            print(f"[score] WARN: reward computation failed for batch ending at idx={idx}: {e}")
            rewards = [{"VQ": -1.0, "MQ": -1.0, "TA": -1.0, "Overall": -1.0}] * len(batch_paths)

        for i, bi in enumerate(batch_indices):
            vq = float(rewards[i]["VQ"])
            mq = float(rewards[i]["MQ"])
            ta = float(rewards[i]["TA"])
            overall = float(rewards[i]["Overall"])
            df.at[bi, "reward_VQ"] = vq
            df.at[bi, "reward_MQ"] = mq
            df.at[bi, "reward_TA"] = ta
            df.at[bi, "reward_Overall"] = overall
            df.at[bi, "reward_composite"] = (
                args.vq_coef * vq + args.mq_coef * mq + args.ta_coef * ta
            )
        # Periodic flush so we can resume / inspect mid-run.
        df.to_csv(output_csv, index=False)
        batch_paths.clear()
        batch_prompts.clear()
        batch_indices.clear()

    df.to_csv(output_csv, index=False)
    print(f"[score] done. {len(df)} videos scored, saved to {output_csv}")
    print(f"[score] composite weights saved to {coefs_path}")


if __name__ == "__main__":
    main()
