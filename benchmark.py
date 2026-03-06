#!/usr/bin/env python3
"""
Benchmark multiple Whisper models on Korean telephonic audio.

Multi-GPU inference (1 process per GPU) for maximum throughput.
Evaluates models on D01, D02, D03, D04 categories and reports:
- CER/WER per category (D01, D02, D03, D04)
- Total CER/WER

Usage:
    python benchmark_whisper.py --num-samples 1000 --batch-size 32
"""

import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

import argparse
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
import evaluate


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = [
#    "/workspace/whisper-large-v3-turbo-korean-ft/checkpoint-4000",
#   "/workspace/whisper-large-v3-turbo-trial1/final",
#    "/workspace/whisper-medium-ft",
    "/workspace/whisper-medium-ft-2",
]

DATA_DIR = "/workspace/data/2.Validation"
CATEGORIES = ["D01", "D02", "D03", "D04"]
TARGET_SR = 16000
DEFAULT_NUM_SAMPLES = None  # None = use all samples
DEFAULT_BATCH_SIZE = 32     # Per-GPU batch size
DEFAULT_NUM_WORKERS = 200   # For manifest/transcript building
DEFAULT_NUM_GPUS = None     # None = use all visible GPUs


# ============================================================
# DATA DISCOVERY
# ============================================================

def find_audio_files_by_category(
    data_dir: Path,
    num_samples_per_category: int = None
) -> Dict[str, List[Tuple[str, str]]]:
    """Find WAV files and pair with TXT transcriptions, organized by category."""
    import random
    random.seed(42)

    category_pairs: Dict[str, List[Tuple[str, str]]] = {}

    for category in CATEGORIES:
        print(f"  Scanning {category}...")

        # Find WAV files in 원천데이터_230316/{category}/
        wav_dir = data_dir / "원천데이터_230316" / category

        if wav_dir.exists():
            wav_files = list(wav_dir.glob("**/*.wav"))
            wav_files.extend(list(wav_dir.glob("**/*.WAV")))
        else:
            # Try alternative patterns
            wav_files = list(data_dir.glob(f"**/원천데이터*/**/{category}/**/*.wav"))
            wav_files.extend(list(data_dir.glob(f"**/원천데이터*/**/{category}/**/*.WAV")))

        print(f"    Found {len(wav_files):,} WAV files")

        # Shuffle and optionally limit
        random.shuffle(wav_files)
        if num_samples_per_category is not None:
            wav_files = wav_files[:num_samples_per_category]
            print(f"    Using {len(wav_files):,} samples (limited)")
        else:
            print(f"    Using ALL {len(wav_files):,} samples")

        # Pair with transcriptions
        pairs: List[Tuple[str, str]] = []
        missing = 0

        for wav_path in wav_files:
            wav_str = str(wav_path)

            # Derive txt path: 원천데이터_230316 -> 라벨링데이터_230316
            txt_str = wav_str.replace("원천데이터_230316", "라벨링데이터_230316")
            txt_str = txt_str.replace(".wav", ".txt").replace(".WAV", ".txt")
            txt_path = Path(txt_str)

            if txt_path.exists():
                pairs.append((wav_str, txt_str))
            else:
                missing += 1

        if missing > 0:
            print(f"    Warning: {missing} files missing transcriptions")

        print(f"    Paired {len(pairs):,} files")
        category_pairs[category] = pairs

    return category_pairs


# ============================================================
# MANIFEST BUILDING (NO AUDIO IN RAM)
# ============================================================

def _load_transcription(item: Tuple[int, str, str, str]) -> Dict[str, Any]:
    """Worker: read a transcription file."""
    idx, category, wav_path, txt_path = item
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            transcription = f.read().strip()
        return {
            "idx": idx,
            "category": category,
            "wav_path": wav_path,
            "transcription": transcription,
            "status": "success",
        }
    except Exception as e:
        return {
            "idx": idx,
            "category": category,
            "wav_path": wav_path,
            "transcription": "",
            "status": "error",
            "error": str(e),
        }


def build_manifest_parallel(
    category_pairs: Dict[str, List[Tuple[str, str]]],
    num_workers: int
) -> List[Dict[str, Any]]:
    """
    Build a manifest of samples without loading audio into memory.
    Each sample: {idx, category, wav_path, transcription}
    """
    all_items: List[Tuple[int, str, str, str]] = []
    idx = 0
    for category, pairs in category_pairs.items():
        for wav_path, txt_path in pairs:
            all_items.append((idx, category, wav_path, txt_path))
            idx += 1

    print(f"\nBuilding manifest for {len(all_items):,} samples with {num_workers} workers...")

    ctx = mp.get_context("spawn")
    results: List[Dict[str, Any]] = []
    with ctx.Pool(num_workers) as pool:
        for r in tqdm(pool.imap(_load_transcription, all_items), total=len(all_items), desc="Manifest"):
            if r["status"] == "success":
                # Drop empty references here (optional); keep them if you want to track
                if r["transcription"].strip():
                    results.append(r)

    # Count per category
    cat_counts = defaultdict(int)
    for r in results:
        cat_counts[r["category"]] += 1
    for cat in CATEGORIES:
        print(f"  {cat}: {cat_counts[cat]:,} samples (non-empty refs)")

    return results


# ============================================================
# MULTI-GPU INFERENCE (1 process per GPU)
# ============================================================

def _load_audio_16k_np(wav_path: str) -> np.ndarray:
    """
    Fast, dependency-light audio loader using torchaudio.
    Returns mono float32 numpy array at 16 kHz.
    """
    import torchaudio
    import torchaudio.functional as AF

    wav, sr = torchaudio.load(wav_path)  # [C, T], float32
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)  # [T]
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    return wav.cpu().numpy().astype(np.float32, copy=False)


def _infer_worker(
    rank: int,
    world_size: int,
    model_path: str,
    batch_size: int,
    shard: List[Dict[str, Any]],
    out_q: mp.Queue,
):
    """
    One GPU worker: loads model onto cuda:{rank}, runs inference for its shard,
    and streams results back via out_q in batch-sized chunks.
    """
    # Reduce CPU thread contention (important with 8 GPU procs)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(f"cuda:{rank}")

    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    # Optional: try to enable flash attention if your env supports it; fallback gracefully.
    def _load_model():
        try:
            return WhisperForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            return WhisperForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            )

    processor = WhisperProcessor.from_pretrained(model_path)
    model = _load_model().to(device)
    model.eval()

    # Configure generation
    model.generation_config.language = "ko"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Inference loop
    with torch.inference_mode():
        for i in range(0, len(shard), batch_size):
            batch_samples = shard[i:i + batch_size]

            # Load audio on-demand (avoids huge IPC / RAM)
            batch_audio = []
            ok_samples = []
            for s in batch_samples:
                try:
                    a = _load_audio_16k_np(s["wav_path"])
                    batch_audio.append(a)
                    ok_samples.append(s)
                except Exception as e:
                    # Skip problematic audio files but keep going
                    # (If you prefer, you can emit an "error" result here.)
                    continue

            if not ok_samples:
                continue

            inputs = processor(
                batch_audio,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True,
            )
            input_features = inputs.input_features.to(device, dtype=torch.float16)

            predicted_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=1,
            )
            predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            chunk = []
            for s, pred in zip(ok_samples, predictions):
                chunk.append({
                    "idx": s["idx"],
                    "category": s["category"],
                    "reference": s["transcription"],
                    "prediction": pred,
                })

            out_q.put(chunk)

    # Signal completion for this worker
    out_q.put(None)

    # Cleanup
    del model
    del processor
    torch.cuda.empty_cache()


def run_inference_multi_gpu(
    samples: List[Dict[str, Any]],
    model_path: str,
    batch_size: int,
    num_gpus: int,
) -> List[Dict[str, Any]]:
    """
    Multi-GPU inference by sharding samples and running 1 process per GPU.
    batch_size is PER-GPU batch size.
    """
    assert num_gpus >= 1, "num_gpus must be >= 1"

    # Shard samples round-robin for load balance
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_gpus)]
    for i, s in enumerate(samples):
        shards[i % num_gpus].append(s)

    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue(maxsize=8 * 8)  # buffer some chunks

    procs = []
    for rank in range(num_gpus):
        p = ctx.Process(
            target=_infer_worker,
            args=(rank, num_gpus, model_path, batch_size, shards[rank], out_q),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # Collect streamed chunks until all workers send None
    done = 0
    results: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(samples), desc="Inference (all GPUs)", leave=False)

    try:
        while done < num_gpus:
            msg = out_q.get()
            if msg is None:
                done += 1
                continue
            # msg is a chunk (list of results)
            results.extend(msg)
            pbar.update(len(msg))
    finally:
        pbar.close()
        for p in procs:
            p.join()

    return results


# ============================================================
# METRICS
# ============================================================

def compute_metrics_by_category(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute CER and WER metrics by category and total."""
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    metrics: Dict[str, Dict[str, float]] = {}
    all_refs: List[str] = []
    all_preds: List[str] = []

    for category in CATEGORIES:
        cat_results = by_category.get(category, [])
        if not cat_results:
            metrics[category] = {"cer": None, "wer": None, "count": 0}
            continue

        refs = [r["reference"] for r in cat_results]
        preds = [r["prediction"] for r in cat_results]

        valid = [(p, r) for p, r in zip(preds, refs) if r.strip()]
        if not valid:
            metrics[category] = {"cer": None, "wer": None, "count": 0}
            continue

        preds_valid, refs_valid = zip(*valid)

        cer = cer_metric.compute(predictions=preds_valid, references=refs_valid) * 100
        wer = wer_metric.compute(predictions=preds_valid, references=refs_valid) * 100

        metrics[category] = {"cer": cer, "wer": wer, "count": len(valid)}

        all_refs.extend(refs_valid)
        all_preds.extend(preds_valid)

    if all_refs:
        total_cer = cer_metric.compute(predictions=all_preds, references=all_refs) * 100
        total_wer = wer_metric.compute(predictions=all_preds, references=all_refs) * 100
        metrics["TOTAL"] = {"cer": total_cer, "wer": total_wer, "count": len(all_refs)}
    else:
        metrics["TOTAL"] = {"cer": None, "wer": None, "count": 0}

    return metrics


# ============================================================
# REPORTING
# ============================================================

def get_short_model_name(model_path: str) -> str:
    path = Path(model_path)
    name = path.name
    if name in ["final", "checkpoint-4000", "checkpoint-2000"]:
        name = f"{path.parent.name}/{name}"
    return name[:35]


def print_results_table(all_results: Dict[str, Dict[str, Dict[str, float]]]):
    model_paths = list(all_results.keys())
    categories = CATEGORIES + ["TOTAL"]

    short_names = [get_short_model_name(m) for m in model_paths]
    max_name_len = max(len(n) for n in short_names) + 2

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS - CER (%)")
    print("=" * 100)

    header = f"{'Model':<{max_name_len}}"
    for cat in categories:
        header += f"{cat:>12}"
    print(header)
    print("-" * 100)

    for model_path, short_name in zip(model_paths, short_names):
        metrics = all_results[model_path]
        row = f"{short_name:<{max_name_len}}"
        for cat in categories:
            cer = metrics.get(cat, {}).get("cer")
            row += f"{cer:>12.2f}" if cer is not None else f"{'N/A':>12}"
        print(row)

    print("=" * 100)

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS - WER (%)")
    print("=" * 100)

    print(header)
    print("-" * 100)

    for model_path, short_name in zip(model_paths, short_names):
        metrics = all_results[model_path]
        row = f"{short_name:<{max_name_len}}"
        for cat in categories:
            wer = metrics.get(cat, {}).get("wer")
            row += f"{wer:>12.2f}" if wer is not None else f"{'N/A':>12}"
        print(row)

    print("=" * 100)


def save_results_csv(all_results: Dict[str, Dict[str, Dict[str, float]]], output_path: str):
    import csv
    categories = CATEGORIES + ["TOTAL"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Model"]
        for cat in categories:
            header.extend([f"{cat}_CER", f"{cat}_WER", f"{cat}_Count"])
        writer.writerow(header)

        for model_path, metrics in all_results.items():
            short_name = get_short_model_name(model_path)
            row = [short_name]
            for cat in categories:
                cer = metrics.get(cat, {}).get("cer")
                wer = metrics.get(cat, {}).get("wer")
                count = metrics.get(cat, {}).get("count", 0)
                row.extend([
                    f"{cer:.2f}" if cer is not None else "N/A",
                    f"{wer:.2f}" if wer is not None else "N/A",
                    count,
                ])
            writer.writerow(row)

    print(f"\nResults saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisper models (multi-GPU)")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Path to validation data")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples per category (None = use all)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Per-GPU batch size for inference")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="CPU workers for manifest build")
    parser.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS, help="GPUs to use (None = all visible)")
    parser.add_argument("--output-csv", type=str, default="benchmark_results.csv", help="Output CSV file path")
    args = parser.parse_args()

    # Ensure spawn (recommended with CUDA multiprocessing)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    start_time = time.time()

    num_samples_str = "ALL" if args.num_samples is None else f"{args.num_samples:,}"

    visible_gpus = torch.cuda.device_count()
    if visible_gpus == 0:
        print("ERROR: No CUDA GPUs detected.")
        return

    num_gpus = visible_gpus if args.num_gpus is None else min(args.num_gpus, visible_gpus)

    print("=" * 100)
    print("WHISPER MODEL BENCHMARK (MULTI-GPU INFERENCE)")
    print("=" * 100)
    print(f"Data directory:      {args.data_dir}")
    print(f"Categories:          {', '.join(CATEGORIES)}")
    print(f"Samples/category:    {num_samples_str}")
    print(f"Batch size (per GPU):{args.batch_size}")
    print(f"CPU workers:         {args.num_workers}")
    print(f"GPUs visible:        {visible_gpus}")
    print(f"GPUs used:           {num_gpus}")
    print(f"\nModels to evaluate:")
    for m in MODELS:
        exists = "✓" if Path(m).exists() else "✗ NOT FOUND"
        print(f"  {exists} {m}")
    print("=" * 100)

    valid_models = [m for m in MODELS if Path(m).exists()]
    if not valid_models:
        print("\nERROR: No valid models found!")
        return

    print(f"\nFound {len(valid_models)} valid models")

    # Find audio files by category
    print("\nFinding audio files by category...")
    category_pairs = find_audio_files_by_category(Path(args.data_dir), args.num_samples)

    total_files = sum(len(pairs) for pairs in category_pairs.values())
    if total_files == 0:
        print("ERROR: No audio files found!")
        return
    print(f"\nTotal paired files (incl. possibly empty refs): {total_files:,}")

    # Build manifest once (shared across all models; no audio loaded yet)
    manifest = build_manifest_parallel(category_pairs, args.num_workers)
    if not manifest:
        print("ERROR: No usable samples in manifest (non-empty refs).")
        return
    print(f"\nUsable samples in manifest: {len(manifest):,}")

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for i, model_path in enumerate(valid_models):
        model_name = get_short_model_name(model_path)

        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(valid_models)}] Evaluating: {model_name}")
        print(f"{'='*80}")

        try:
            results = run_inference_multi_gpu(
                samples=manifest,
                model_path=model_path,
                batch_size=args.batch_size,
                num_gpus=num_gpus,
            )

            metrics = compute_metrics_by_category(results)
            all_results[model_path] = metrics

            print(f"\n  Results for {model_name}:")
            for cat in CATEGORIES + ["TOTAL"]:
                cer = metrics.get(cat, {}).get("cer")
                wer = metrics.get(cat, {}).get("wer")
                cnt = metrics.get(cat, {}).get("count", 0)
                if cer is not None:
                    print(f"    {cat}: CER = {cer:.2f}% | WER = {wer:.2f}% | N = {cnt:,}")
                else:
                    print(f"    {cat}: N/A | N = {cnt:,}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_path] = {
                cat: {"cer": None, "wer": None, "count": 0}
                for cat in CATEGORIES + ["TOTAL"]
            }

    print_results_table(all_results)
    save_results_csv(all_results, args.output_csv)

    elapsed = time.time() - start_time
    print(f"\nTotal benchmark time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")


if __name__ == "__main__":
    main()

