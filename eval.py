#!/usr/bin/env python3
"""
Evaluate Whisper model on Korean telephonic audio.

Uses:
- 200 CPU cores for audio preprocessing
- 8 H200 GPUs for inference
- Reports CER and WER

Usage:
    python evaluate_whisper.py \
        --model-path /workspace/whisper-medium-ft \
        --data-dir /workspace/data/2.Validation \
        --num-samples 5000 \
        --batch-size 32
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
from typing import List, Tuple, Dict
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm
import evaluate


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL_PATH = "/workspace/whisper-medium-ft"
DEFAULT_DATA_DIR = "/workspace/data/2.Validation"
DEFAULT_NUM_SAMPLES = 5000
DEFAULT_BATCH_SIZE = 32
TARGET_SR = 16000


# ============================================================
# DATA LOADING
# ============================================================

def find_audio_files(data_dir: Path, num_samples: int) -> List[Tuple[str, str]]:
    """Find WAV files and pair with TXT transcriptions."""
    
    print(f"Scanning {data_dir} for WAV files...")
    
    # Find all WAV files
    wav_files = list(data_dir.glob("**/원천데이터*/**/*.wav"))
    wav_files.extend(list(data_dir.glob("**/원천데이터*/**/*.WAV")))
    
    if not wav_files:
        # Try without 원천데이터 pattern
        wav_files = list(data_dir.glob("**/*.wav"))
        wav_files.extend(list(data_dir.glob("**/*.WAV")))
    
    print(f"Found {len(wav_files):,} WAV files")
    
    # Shuffle and limit
    import random
    random.seed(42)
    random.shuffle(wav_files)
    wav_files = wav_files[:num_samples]
    
    print(f"Using {len(wav_files):,} samples for evaluation")
    
    # Pair with transcriptions
    pairs = []
    missing = 0
    
    for wav_path in wav_files:
        wav_str = str(wav_path)
        
        # Derive txt path
        txt_str = wav_str.replace("원천데이터", "라벨링데이터").replace(".wav", ".txt").replace(".WAV", ".txt")
        txt_path = Path(txt_str)
        
        if txt_path.exists():
            pairs.append((wav_str, txt_str))
        else:
            missing += 1
    
    if missing > 0:
        print(f"Warning: {missing} files missing transcriptions")
    
    print(f"Successfully paired {len(pairs):,} files")
    return pairs


def load_and_preprocess_audio(args: Tuple[str, str, int]) -> Dict:
    """Load and preprocess a single audio file."""
    import librosa
    
    wav_path, txt_path, idx = args
    
    try:
        # Load audio
        audio, sr = librosa.load(wav_path, sr=TARGET_SR, res_type='kaiser_fast')
        
        # Load transcription
        with open(txt_path, 'r', encoding='utf-8') as f:
            transcription = f.read().strip()
        
        return {
            "idx": idx,
            "audio": audio,
            "transcription": transcription,
            "status": "success"
        }
    except Exception as e:
        return {
            "idx": idx,
            "status": "error",
            "error": str(e)
        }


def preprocess_audio_parallel(pairs: List[Tuple[str, str]], num_workers: int) -> List[Dict]:
    """Preprocess audio files in parallel."""
    
    print(f"\nPreprocessing {len(pairs):,} audio files with {num_workers} workers...")
    
    args_list = [(wav, txt, i) for i, (wav, txt) in enumerate(pairs)]
    
    ctx = mp.get_context('spawn')
    
    results = []
    with ctx.Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap(load_and_preprocess_audio, args_list),
            total=len(args_list),
            desc="Preprocessing"
        ):
            if result["status"] == "success":
                results.append(result)
    
    print(f"Successfully preprocessed {len(results):,} files")
    return results


# ============================================================
# INFERENCE
# ============================================================

def run_inference_multi_gpu(
    samples: List[Dict],
    model_path: str,
    batch_size: int,
    num_gpus: int
) -> List[Dict]:
    """Run inference using multiple GPUs with data parallelism."""
    
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    print(f"\nLoading model from {model_path}...")
    
    # Load processor
    processor = WhisperProcessor.from_pretrained(model_path)
    
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"  # Automatically distribute across GPUs
    )
    model.eval()
    
    # Configure generation
    model.generation_config.language = "ko"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    print(f"Model loaded. Running inference...")
    
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(samples), batch_size), desc="Inference"):
        batch_samples = samples[i:i + batch_size]
        
        # Prepare batch
        batch_audio = [s["audio"] for s in batch_samples]
        
        # Extract features
        inputs = processor(
            batch_audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True
        )
        
        input_features = inputs.input_features.to(model.device, dtype=torch.float16)
        
        # Generate
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=1,  # Greedy decoding for speed
            )
        
        # Decode
        predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        # Store results
        for sample, pred in zip(batch_samples, predictions):
            results.append({
                "idx": sample["idx"],
                "reference": sample["transcription"],
                "prediction": pred
            })
    
    return results


# ============================================================
# METRICS
# ============================================================

def compute_metrics(results: List[Dict]) -> Dict:
    """Compute CER and WER metrics."""
    
    print("\nComputing metrics...")
    
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    references = [r["reference"] for r in results]
    predictions = [r["prediction"] for r in results]
    
    # Filter empty references
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    
    if not valid_pairs:
        return {"cer": 0.0, "wer": 0.0}
    
    predictions_filtered, references_filtered = zip(*valid_pairs)
    
    cer = cer_metric.compute(predictions=predictions_filtered, references=references_filtered)
    wer = wer_metric.compute(predictions=predictions_filtered, references=references_filtered)
    
    return {
        "cer": cer * 100,
        "wer": wer * 100,
        "num_samples": len(valid_pairs)
    }


def print_sample_predictions(results: List[Dict], num_samples: int = 10):
    """Print sample predictions for inspection."""
    
    print(f"\n{'='*70}")
    print("SAMPLE PREDICTIONS")
    print('='*70)
    
    import random
    random.seed(42)
    samples = random.sample(results, min(num_samples, len(results)))
    
    for i, sample in enumerate(samples):
        print(f"\n[{i+1}] Reference:  {sample['reference'][:100]}...")
        print(f"    Prediction: {sample['prediction'][:100]}...")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper model")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to finetuned model")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Path to validation data")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="Number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=200,
                        help="Number of CPU workers for preprocessing")
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("=" * 70)
    print("WHISPER EVALUATION")
    print("=" * 70)
    print(f"Model:       {args.model_path}")
    print(f"Data:        {args.data_dir}")
    print(f"Samples:     {args.num_samples:,}")
    print(f"Batch size:  {args.batch_size}")
    print(f"CPU workers: {args.num_workers}")
    print(f"GPUs:        {torch.cuda.device_count()}")
    print("=" * 70)
    
    # Find audio files
    pairs = find_audio_files(Path(args.data_dir), args.num_samples)
    
    if not pairs:
        print("ERROR: No audio files found!")
        return
    
    # Preprocess audio in parallel
    samples = preprocess_audio_parallel(pairs, args.num_workers)
    
    if not samples:
        print("ERROR: No samples preprocessed successfully!")
        return
    
    # Run inference
    results = run_inference_multi_gpu(
        samples,
        args.model_path,
        args.batch_size,
        torch.cuda.device_count()
    )
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Print sample predictions
    print_sample_predictions(results)
    
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Model:           {args.model_path}")
    print(f"Samples:         {metrics['num_samples']:,}")
    print(f"CER:             {metrics['cer']:.2f}%")
    print(f"WER:             {metrics['wer']:.2f}%")
    print(f"Total time:      {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Speed:           {metrics['num_samples']/elapsed:.1f} samples/sec")
    print("=" * 70)


if __name__ == "__main__":
    main()
