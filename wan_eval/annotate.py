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

Pair construction (within each prompt's K seeds):
  - default (--pairs_per_prompt 1): randomly pick ONE pair per prompt
    -> 20 prompts x 1 pair = 20 rows total (minimal annotation workload)
  - --pairs_per_prompt N: pick N distinct random pairs per prompt (capped at C(K,2))
  - --pairs_per_prompt all: all C(K, 2) combinations per prompt (legacy behaviour)

Usage (run from inside VideoAlign/wan_eval/):
    python annotate.py \
        --reward_scores outputs/reward_scores.csv \
        --output_dir outputs
        # default: 1 pair per prompt -> ~20 pairs for the default 20 prompts
"""
from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pointwise + pairwise annotation templates.")
    p.add_argument("--reward_scores", default="outputs/reward_scores.csv",
                   help="CSV from Phase B; we only need prompt_id/video_id/prompt/video_path columns. "
                        "Default: outputs/reward_scores.csv")
    p.add_argument("--output_dir", default=None,
                   help="Where to write the two CSVs. Defaults to dir of --reward_scores.")
    p.add_argument("--pairs_per_prompt", default="1",
                   help="Pairs per prompt. Either an integer (>=1) or 'all' for C(K,2). Default '1'.")
    p.add_argument("--pair_sample", type=int, default=None,
                   help="(Optional) Global cap on the total number of pairs after per-prompt selection.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for pair sampling.")
    return p.parse_args()


def _select_pairs_for_group(ids: list[str], pairs_per_prompt: str, rng: random.Random) -> list[tuple[str, str]]:
    """Pick the pairs for one prompt according to `--pairs_per_prompt`."""
    all_pairs = list(itertools.combinations(ids, 2))
    if pairs_per_prompt == "all":
        return all_pairs
    try:
        n = int(pairs_per_prompt)
    except ValueError as e:
        raise ValueError(f"--pairs_per_prompt must be 'all' or an integer, got '{pairs_per_prompt}'") from e
    if n <= 0:
        raise ValueError("--pairs_per_prompt must be >= 1 (or 'all').")
    if n >= len(all_pairs):
        return all_pairs
    return rng.sample(all_pairs, n)


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
    rng = random.Random(args.seed)
    pair_rows: list[dict] = []
    pair_id = 0
    for prompt_id, group in df.groupby("prompt_id"):
        ids = group["video_id"].tolist()
        paths = dict(zip(group["video_id"], group["video_path"]))
        prompt = group["prompt"].iloc[0]
        for a, b in _select_pairs_for_group(ids, args.pairs_per_prompt, rng):
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
