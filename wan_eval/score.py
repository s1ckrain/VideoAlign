"""
Phase B: score generated videos with VideoReward (KwaiVGI/VideoReward).

Input : {output_dir}/videos_meta.csv     (from Phase A)
Output: {output_dir}/reward_scores.csv   (videos_meta + reward_{VQ,MQ,TA,Overall})

Usage (run from inside VideoAlign/wan_eval/):
    python score.py \
        --videos_meta outputs/videos_meta.csv \
        --gpu 0

    # --videoalign_dir defaults to ".." (i.e. the VideoAlign repo root) since
    # this script lives at VideoAlign/wan_eval/score.py.
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
    for col in ("reward_VQ", "reward_MQ", "reward_TA", "reward_Overall"):
        if col not in df.columns:
            df[col] = float("nan")

    output_csv = Path(args.output_csv) if args.output_csv else meta_parent / "reward_scores.csv"

    # Mini-batch loop (same pattern as VideoAlign's eval_videogen_rewardbench.py).
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
        with torch.no_grad():
            rewards = inferencer.reward(batch_paths, batch_prompts)
        for i, bi in enumerate(batch_indices):
            df.at[bi, "reward_VQ"] = rewards[i]["VQ"]
            df.at[bi, "reward_MQ"] = rewards[i]["MQ"]
            df.at[bi, "reward_TA"] = rewards[i]["TA"]
            df.at[bi, "reward_Overall"] = rewards[i]["Overall"]
        # Periodic flush so we can resume / inspect mid-run.
        df.to_csv(output_csv, index=False)
        batch_paths.clear()
        batch_prompts.clear()
        batch_indices.clear()

    df.to_csv(output_csv, index=False)
    print(f"[score] done. {len(df)} videos scored, saved to {output_csv}")


if __name__ == "__main__":
    main()
