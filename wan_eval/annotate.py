"""
Phase C: build human-annotation templates from reward_scores.csv.

Outputs two CSVs (Excel/WPS friendly) for the annotators to fill in:

1) human_pointwise.csv
   columns: video_id, prompt, video_path, human_VQ, human_MQ, human_TA
   - human_* fields should be filled with an integer 1..5 (or 1..7) per dim.

2) human_pairwise.csv
   columns: pair_id, prompt, video_A_id, video_B_id, video_A_path, video_B_path,
            human_VQ, human_MQ, human_TA
   - human_* fields should be filled with one of: 'A' / 'B' / 'same'

Pairs are formed WITHIN each prompt: all C(K, 2) combinations of the K seeds.

Usage (run from inside VideoAlign/wan_eval/):
    python annotate.py \
        --reward_scores outputs/reward_scores.csv \
        --output_dir outputs \
        [--pair_sample 100]            # sample at most 100 pairs total
"""
from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pointwise + pairwise annotation templates.")
    p.add_argument("--reward_scores", required=True,
                   help="CSV from Phase B; we only need prompt_id/video_id/prompt/video_path columns.")
    p.add_argument("--output_dir", default=None,
                   help="Where to write the two CSVs. Defaults to dir of --reward_scores.")
    p.add_argument("--pair_sample", type=int, default=None,
                   help="If set, randomly sample at most N pairs (uniform across prompts).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for pair sampling.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.reward_scores).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else src.parent

    df = pd.read_csv(src)
    for col in ("video_id", "prompt_id", "prompt", "video_path"):
        if col not in df.columns:
            raise ValueError(f"`{col}` column missing from {src}")

    # ---- pointwise template -------------------------------------------------
    pw = df[["video_id", "prompt", "video_path"]].copy()
    for c in ("human_VQ", "human_MQ", "human_TA"):
        pw[c] = ""   # integer 1..5 (annotator fills)
    pw_path = out_dir / "human_pointwise.csv"
    pw.to_csv(pw_path, index=False)
    print(f"[anno] pointwise template -> {pw_path}  ({len(pw)} rows)")

    # ---- pairwise template --------------------------------------------------
    pair_rows = []
    pair_id = 0
    for prompt_id, group in df.groupby("prompt_id"):
        ids = group["video_id"].tolist()
        paths = dict(zip(group["video_id"], group["video_path"]))
        prompt = group["prompt"].iloc[0]
        for a, b in itertools.combinations(ids, 2):
            pair_rows.append({
                "pair_id": pair_id,
                "prompt_id": int(prompt_id),
                "prompt": prompt,
                "video_A_id": a,
                "video_B_id": b,
                "video_A_path": paths[a],
                "video_B_path": paths[b],
                "human_VQ": "",   # 'A' / 'B' / 'same'
                "human_MQ": "",
                "human_TA": "",
            })
            pair_id += 1

    if args.pair_sample and args.pair_sample < len(pair_rows):
        rng = random.Random(args.seed)
        pair_rows = rng.sample(pair_rows, args.pair_sample)
        for i, r in enumerate(pair_rows):
            r["pair_id"] = i

    pair_df = pd.DataFrame(pair_rows)
    pair_path = out_dir / "human_pairwise.csv"
    pair_df.to_csv(pair_path, index=False)
    print(f"[anno] pairwise template -> {pair_path}  ({len(pair_df)} rows)")

    print("\n[anno] How to annotate:")
    print(f"  - {pw_path.name}: fill human_VQ/MQ/TA with INTEGER 1..5 (or 1..7).")
    print(f"  - {pair_path.name}: fill human_VQ/MQ/TA with 'A' / 'B' / 'same'.")


if __name__ == "__main__":
    main()
