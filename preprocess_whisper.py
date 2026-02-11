#!/usr/bin/env python3
"""
High-Performance Whisper Preprocessing Script

Optimized for high-core-count machines by avoiding multiprocessing IPC overhead.
Each worker saves results directly to disk instead of returning large arrays.

Usage:
    python preprocess_whisper_fast.py \
        --input-dir /workspace/data \
        --output-dir /workspace/data/processed \
        --num-workers 200
"""

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import partial
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Prevent numpy/scipy internal threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def process_and_save_chunk(args: Tuple[List[Tuple[str, str, str]], str, str, int, str, int, bool]) -> Dict:
    """
    Process a chunk of files and save directly to disk.
    This avoids IPC overhead by not returning large arrays.
    
    Args:
        args: Tuple containing:
            - file_list: List of (file_path, transcription, file_id) tuples
            - output_dir: Base output directory
            - split: 'train' or 'val'
            - chunk_id: Unique chunk identifier
            - model_name: Whisper model name for feature extractor
            - target_sr: Target sample rate
            - use_fp16: Whether to use float16
    
    Returns:
        Dict with success/error counts
    """
    import librosa
    from transformers import WhisperFeatureExtractor, WhisperTokenizer
    from datasets import Dataset
    
    file_list, output_dir, split, chunk_id, model_name, target_sr, use_fp16 = args
    
    # Initialize feature extractor and tokenizer in this process
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="ko", task="transcribe")
    
    successful_data = []
    error_count = 0
    total_duration = 0.0
    
    for file_path, transcription, file_id in file_list:
        try:
            # Load and resample audio
            audio, sr = librosa.load(file_path, sr=target_sr, res_type='kaiser_fast')
            
            # Skip very short audio
            if len(audio) < 1600:
                continue
            
            # Compute mel spectrogram
            input_features = feature_extractor(
                audio,
                sampling_rate=target_sr,
                return_tensors="np"
            ).input_features[0]
            
            if use_fp16:
                input_features = input_features.astype(np.float16)
            
            # Tokenize transcription
            labels = tokenizer(transcription).input_ids
            
            duration = len(audio) / target_sr
            total_duration += duration
            
            successful_data.append({
                "input_features": input_features,
                "labels": labels,
                "file_id": file_id,
                "duration": duration,
            })
            
        except Exception as e:
            error_count += 1
    
    # Save chunk directly to disk
    if successful_data:
        output_path = Path(output_dir) / split / f"chunk_{chunk_id:06d}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = Dataset.from_dict({
            "input_features": [d["input_features"] for d in successful_data],
            "labels": [d["labels"] for d in successful_data],
            "file_id": [d["file_id"] for d in successful_data],
            "duration": [d["duration"] for d in successful_data],
        })
        dataset.save_to_disk(str(output_path))
    
    return {
        "chunk_id": chunk_id,
        "successful": len(successful_data),
        "errors": error_count,
        "duration": total_duration,
    }


def load_file_list(input_dir: Path, num_workers: int = 64) -> pd.DataFrame:
    """Load WAV files and pair with TXT transcriptions using parallel I/O."""
    
    print(f"Scanning {input_dir} for WAV files...")
    wav_files = list(input_dir.glob("**/*.wav"))
    wav_files.extend(list(input_dir.glob("**/*.WAV")))
    
    if not wav_files:
        raise ValueError(f"No WAV files found in {input_dir}")
    
    print(f"Found {len(wav_files):,} WAV files")
    print(f"Pairing with TXT transcription files using {num_workers} threads...")
    
    def process_single_wav(args):
        """Process a single WAV file - find and read its TXT."""
        i, wav_path = args
        wav_str = str(wav_path)
        
        # Derive txt path: 원천데이터 -> 라벨링데이터, .wav -> .txt
        txt_str = wav_str.replace("원천데이터", "라벨링데이터").replace(".wav", ".txt").replace(".WAV", ".txt")
        txt_path = Path(txt_str)
        
        if txt_path.exists():
            try:
                transcription = txt_path.read_text(encoding="utf-8").strip()
                return {
                    "file_path": wav_str,
                    "transcription": transcription,
                    "file_id": f"sample_{i:08d}"
                }
            except:
                return None
        return None
    
    # Process in parallel using threads (I/O bound task)
    from concurrent.futures import ThreadPoolExecutor
    
    args_list = [(i, wav_path) for i, wav_path in enumerate(wav_files)]
    
    data = []
    missing_txt_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_wav, args_list),
            total=len(args_list),
            desc="Scanning"
        ))
    
    for r in results:
        if r is not None:
            data.append(r)
        else:
            missing_txt_count += 1
    
    if missing_txt_count > 0:
        print(f"⚠️  Warning: {missing_txt_count:,} WAV files missing TXT files")
    
    df = pd.DataFrame(data)
    print(f"Successfully paired {len(df):,} files")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Fast Whisper preprocessing")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Files per chunk (each chunk processed by one worker)")
    parser.add_argument("--model-name", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.01)
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FAST WHISPER PREPROCESSING")
    print("=" * 70)
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"Workers:     {args.num_workers}")
    print(f"Chunk size:  {args.chunk_size}")
    print(f"CPUs:        {mp.cpu_count()}")
    print("=" * 70)
    
    # Load file list with parallel I/O
    df = load_file_list(input_dir, num_workers=min(128, args.num_workers))
    
    # Shuffle and split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = int(len(df) * args.val_split)
    
    val_df = df.head(val_size)
    train_df = df.tail(len(df) - val_size)
    
    print(f"\nSplit: {len(train_df):,} train, {len(val_df):,} val")
    
    # Pre-cache feature extractor
    print("\nCaching feature extractor...")
    from transformers import WhisperFeatureExtractor
    _ = WhisperFeatureExtractor.from_pretrained(args.model_name)
    print("✓ Cached")
    
    start_time = time.time()
    
    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        if len(split_df) == 0:
            continue
            
        print(f"\n{'=' * 70}")
        print(f"Processing {split_name}: {len(split_df):,} files")
        print("=" * 70)
        
        # Convert to list of tuples
        file_list = list(zip(
            split_df["file_path"].tolist(),
            split_df["transcription"].tolist(),
            split_df["file_id"].tolist()
        ))
        
        # Split into chunks
        chunks = []
        for i in range(0, len(file_list), args.chunk_size):
            chunk = file_list[i:i + args.chunk_size]
            chunk_id = len(chunks)
            chunks.append((
                chunk,
                str(output_dir),
                split_name,
                chunk_id,
                args.model_name,
                args.target_sr,
                not args.no_fp16
            ))
        
        print(f"Created {len(chunks):,} chunks of ~{args.chunk_size} files each")
        
        # Process chunks in parallel
        total_successful = 0
        total_errors = 0
        total_duration = 0.0
        
        # Use spawn to avoid issues with fork
        ctx = mp.get_context('spawn')
        
        with ctx.Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_and_save_chunk, chunks),
                total=len(chunks),
                desc=f"{split_name}",
                unit="chunks"
            ))
        
        for r in results:
            total_successful += r["successful"]
            total_errors += r["errors"]
            total_duration += r["duration"]
        
        print(f"\n{split_name}: {total_successful:,} successful, {total_errors:,} errors")
        print(f"Total duration: {total_duration / 3600:.1f} hours of audio")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed / 3600:.2f} hours")
    print(f"Output: {output_dir}")
    
    # Save metadata
    meta = {
        "total_time_hours": elapsed / 3600,
        "num_workers": args.num_workers,
        "chunk_size": args.chunk_size,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
