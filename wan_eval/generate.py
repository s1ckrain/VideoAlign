"""
Phase A: generate videos with Wan2.1 for reward-model validation.

Layout:
    {output_dir}/
        videos/p0000_s0.mp4, p0000_s1.mp4, ..., p{P-1}_s{S-1}.mp4
        videos_meta.csv  -> (video_id, prompt_id, seed, prompt, video_path)

Usage (run from inside VideoAlign/wan_eval/):
    python generate.py \
        --prompts_file prompts.txt \
        --output_dir outputs \
        --model_name Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
        --num_seeds 4 --gpu 0
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm import tqdm

from diffusers import WanPipeline
from diffusers.utils import export_to_video


# ---- Recommended config presets per model ------------------------------------
# Keyed by an identifier substring; resolve_gen_kwargs() picks a preset by
# fuzzy-matching the substring against --model_name (HF id OR local path).
MODEL_PRESETS = {
    "1.3B": dict(
        height=480, width=832, num_frames=49, fps=15,
        num_inference_steps=30, guidance_scale=5.0,
    ),
    "14B": dict(
        height=480, width=832, num_frames=49, fps=15,
        num_inference_steps=40, guidance_scale=5.0,
    ),
}
_DEFAULT_PRESET_KEY = "1.3B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Wan2.1 videos for reward-model validation.")
    p.add_argument("--prompts_file", required=True,
                   help="Plain .txt (one prompt per line) or .jsonl with {'prompt': ...}")
    p.add_argument("--output_dir", required=True,
                   help="Output root. Videos -> {out}/videos, metadata -> {out}/videos_meta.csv")
    p.add_argument("--model_name", default="/aigc/posttrain/siyuanfu/models/Wan2.1",
                   help="HF model id (e.g. Wan-AI/Wan2.1-T2V-1.3B-Diffusers) "
                        "OR local diffusers path. Default: local 1.3B at "
                        "/aigc/posttrain/siyuanfu/models/Wan2.1")
    p.add_argument("--preset_key", default=None, choices=[None, "1.3B", "14B"],
                   help="Force a generation preset. If omitted, inferred from --model_name "
                        "(substring match on '1.3B' / '14B'); fallback = 1.3B.")
    p.add_argument("--num_seeds", type=int, default=2,
                   help="K = videos per prompt (need >=2 to enable pairwise comparison). "
                        "Default 2 keeps annotation workload small (20 prompts -> 40 videos).")
    p.add_argument("--seed_base", type=int, default=42,
                   help="seed for prompt_i, seed_j = seed_base + j (so each prompt sees the same K seeds).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="Only generate the first N prompts (for smoke test).")
    # Generation overrides (default to MODEL_PRESETS if not set)
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=None,
                   help="MUST satisfy (num_frames - 1) % 4 == 0 for Wan.")
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--negative_prompt", default="",
                   help="Optional negative prompt. Empty disables CFG-neg.")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def load_prompts(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    prompts: list[str] = []
    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                prompts.append(json.loads(line)["prompt"])
    else:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    prompts.append(line)
    return prompts


def resolve_gen_kwargs(args: argparse.Namespace) -> dict:
    # Pick preset: explicit > substring on model_name > fallback (1.3B).
    if args.preset_key is not None:
        preset_key = args.preset_key
    else:
        preset_key = next(
            (k for k in MODEL_PRESETS if k in args.model_name),
            _DEFAULT_PRESET_KEY,
        )
    if preset_key not in MODEL_PRESETS:
        raise ValueError(f"unknown preset_key={preset_key}, available: {list(MODEL_PRESETS)}")
    print(f"[gen] using preset '{preset_key}' for model_name='{args.model_name}'")
    preset = MODEL_PRESETS[preset_key]

    merged = dict(preset)
    for k in ("num_inference_steps", "guidance_scale", "height", "width", "num_frames", "fps"):
        v = getattr(args, k)
        if v is not None:
            merged[k] = v
    if (merged["num_frames"] - 1) % 4 != 0:
        raise ValueError(f"Wan requires (num_frames - 1) % 4 == 0, got {merged['num_frames']}.")
    return merged


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"--gpu={args.gpu} out of range (have {torch.cuda.device_count()} devices)")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    gen_kwargs = resolve_gen_kwargs(args)
    fps = gen_kwargs.pop("fps")
    print(f"[gen] device={device} dtype={dtype} gen_kwargs={gen_kwargs} fps={fps}")

    out_dir = Path(args.output_dir).resolve()
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = out_dir / "videos_meta.csv"

    prompts = load_prompts(args.prompts_file)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    print(f"[gen] {len(prompts)} prompts x {args.num_seeds} seeds = {len(prompts) * args.num_seeds} videos")

    print(f"[gen] loading {args.model_name} ...")
    pipe = WanPipeline.from_pretrained(args.model_name, torch_dtype=dtype)
    pipe.to(device)
    # Hard-disable progress bars inside the pipeline so the outer tqdm is readable.
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    # Resume support: skip videos that already exist on disk.
    existing_rows: list[dict] = []
    if meta_csv.exists():
        with open(meta_csv, "r", newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        existing_ids = {r["video_id"] for r in existing_rows}
        print(f"[gen] resuming: {len(existing_ids)} videos already in {meta_csv.name}")
    else:
        existing_ids = set()

    meta_rows: list[dict] = list(existing_rows)
    pbar = tqdm(total=len(prompts) * args.num_seeds, desc="Generating")
    for prompt_id, prompt in enumerate(prompts):
        for seed_offset in range(args.num_seeds):
            video_id = f"p{prompt_id:04d}_s{seed_offset}"
            video_filename = f"{video_id}.mp4"
            video_path = videos_dir / video_filename

            if video_id in existing_ids and video_path.exists():
                pbar.update(1)
                continue

            seed = args.seed_base + seed_offset
            generator = torch.Generator(device=device).manual_seed(seed)
            output = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt or None,
                generator=generator,
                **gen_kwargs,
            )
            frames = output.frames[0]
            export_to_video(frames, str(video_path), fps=fps)

            meta_rows.append({
                "video_id": video_id,
                "prompt_id": prompt_id,
                "seed_offset": seed_offset,
                "seed": seed,
                "prompt": prompt,
                "video_path": str(video_path.relative_to(out_dir)),
            })
            # Flush metadata after each video so a crash doesn't lose progress.
            with open(meta_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
                w.writeheader()
                w.writerows(meta_rows)
            pbar.update(1)
    pbar.close()
    print(f"[gen] done. {len(meta_rows)} videos in {videos_dir}, metadata at {meta_csv}")


if __name__ == "__main__":
    main()
