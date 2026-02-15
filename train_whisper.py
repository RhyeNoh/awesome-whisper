#!/usr/bin/env python3
"""
Full Finetune Whisper Large-v3 for Korean Telephonic Audio

Optimized for 8x H200 GPUs (143GB each)
"""

# === MUST BE BEFORE ANY OTHER IMPORTS ===
import warnings
import logging
import os

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set environment variables before importing transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Configure logging before importing transformers
logging.basicConfig(level=logging.ERROR)
for logger_name in [
    "transformers",
    "transformers.generation",
    "transformers.modeling_utils", 
    "transformers.tokenization_utils_base",
    "transformers.generation_utils",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# === NOW IMPORT EVERYTHING ELSE ===
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
import evaluate

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
MODEL_NAME = "openai/whisper-large-v3-turbo"
DATASET_DIR = "/workspace/data/processed"
OUTPUT_DIR = "/workspace/whisper-large-v3-turbo-korean-ft"

# Language
LANGUAGE = "ko"
TASK = "transcribe"

# ============================================================
# HYPERPARAMETERS - Optimized for 8x H200 (143GB each)
# ============================================================

# Learning rate
LEARNING_RATE = 5e-5              # Conservative for full finetune (worked well before)
LR_SCHEDULER = "cosine"           # Linear decay
WARMUP_RATIO = 0.05               # 3% warmup

# Batch size (per GPU)
PER_DEVICE_TRAIN_BATCH_SIZE = 256
PER_DEVICE_EVAL_BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 1
# Effective batch = 256 * 1 * 8 GPUs = 2048

# Training duration
NUM_TRAIN_EPOCHS = 5              # Full 5 epochs
MAX_STEPS = -1

# Regularization
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LABEL_SMOOTHING = 0.0             # No label smoothing (was causing issues)

# Evaluation & Saving
EVAL_STEPS = 500                  # Evaluate every 500 steps
SAVE_STEPS = 500                  # Save checkpoint every 500 steps
SAVE_TOTAL_LIMIT = 5
LOGGING_STEPS = 50
MAX_EVAL_SAMPLES = 4000

# Early stopping
EARLY_STOPPING_PATIENCE = 10      # Patient early stopping
EARLY_STOPPING_THRESHOLD = 0.001

# Generation (for evaluation)
GENERATION_MAX_LENGTH = 225
GENERATION_NUM_BEAMS = 1          # Greedy decoding for speed during eval


# ============================================================
# DATA COLLATOR
# ============================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads the inputs and labels.
    """
    processor: Any
    decoder_start_token_id: int
    input_padding: str = "longest"
    label_padding: str = "longest"
    max_label_length: Optional[int] = 448  # Whisper's max

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = []
        label_features = []
        
        for feature in features:
            # Handle input features - convert to float32 first (required by feature extractor padding)
            input_feat = feature["input_features"]
            if isinstance(input_feat, np.ndarray):
                input_feat = input_feat.astype(np.float32)
            input_features.append({"input_features": input_feat})
            
            # Handle labels - TRUNCATE if too long
            labels = list(feature["labels"])
            if self.max_label_length and len(labels) > self.max_label_length:
                labels = labels[:self.max_label_length]
            label_features.append({"input_ids": labels})

        # Pad inputs
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt"
        )
        
        # Convert input_features to bfloat16 to match model dtype
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.label_padding,
            return_tensors="pt"
        )
        
        labels = labels_batch["input_ids"]
        attention_mask = labels_batch["attention_mask"]
        
        # Create decoder_input_ids: shift labels right, prepend decoder_start_token
        # Replace padding with pad_token_id for decoder_input_ids (not -100)
        pad_token_id = self.processor.tokenizer.pad_token_id
        
        decoder_input_ids = labels.clone()
        # Shift right: move everything one position to the right
        decoder_input_ids[:, 1:] = labels[:, :-1]
        decoder_input_ids[:, 0] = self.decoder_start_token_id
        # Keep padding as pad_token_id in decoder_input_ids
        decoder_input_ids = decoder_input_ids.masked_fill(attention_mask.eq(0), pad_token_id)

        # For labels, replace padding with -100 to ignore in loss
        labels = labels.masked_fill(attention_mask.eq(0), -100)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch


# ============================================================
# METRICS
# ============================================================

def create_compute_metrics(processor):
    """Create compute_metrics function with processor in closure."""
    
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad token id
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        
        # Decode predictions and labels
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Filter out empty strings to avoid division by zero
        valid_pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        
        if not valid_pairs:
            return {"cer": 0.0, "wer": 0.0}
        
        pred_filtered, label_filtered = zip(*valid_pairs)
        
        # Compute metrics
        cer = cer_metric.compute(predictions=pred_filtered, references=label_filtered)
        wer = wer_metric.compute(predictions=pred_filtered, references=label_filtered)
        
        return {
            "cer": round(cer * 100, 2),   # As percentage
            "wer": round(wer * 100, 2),   # As percentage
        }
    
    return compute_metrics


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset_from_chunks(dataset_dir: Path, split: str):
    """Load and concatenate all chunks for a split."""
    
    split_dir = dataset_dir / split
    chunk_dirs = sorted(split_dir.glob("chunk_*"))
    
    if not chunk_dirs:
        raise ValueError(f"No chunks found in {split_dir}")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(f"Loading {len(chunk_dirs)} {split} chunks...")
    
    # Load chunks in parallel for speed
    from concurrent.futures import ThreadPoolExecutor
    
    def load_chunk(chunk_dir):
        try:
            return load_from_disk(str(chunk_dir))
        except Exception as e:
            print(f"Warning: Failed to load {chunk_dir}: {e}")
            return None
    
    # Use threads for I/O-bound loading
    num_threads = min(64, len(chunk_dirs))
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        datasets = list(executor.map(load_chunk, chunk_dirs))
    
    # Filter out failed loads
    datasets = [ds for ds in datasets if ds is not None]
    
    if not datasets:
        raise ValueError(f"No valid chunks loaded for {split}")
    
    combined = concatenate_datasets(datasets)
    
    if local_rank == 0:
        print(f"Loaded {split}: {len(combined):,} samples")
    
    return combined


def prepare_dataset(dataset, processor):
    """Dataset is already preprocessed with labels and filtered for length."""
    # Labels are already tokenized and filtered during preprocessing
    required_columns = ["input_features", "labels"]
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Dataset missing required column: {col}")
    return dataset


# ============================================================
# MAIN
# ============================================================

def main():
    # Set up distributed training info
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    is_main_process = local_rank == 0
    
    if is_main_process:
        print("=" * 70)
        print("WHISPER LARGE-V3 FULL FINETUNE")
        print("=" * 70)
        print(f"Model:           {MODEL_NAME}")
        print(f"Dataset:         {DATASET_DIR}")
        print(f"Output:          {OUTPUT_DIR}")
        print(f"GPUs:            {world_size}")
        print(f"Per-GPU batch:   {PER_DEVICE_TRAIN_BATCH_SIZE}")
        print(f"Grad accum:      {GRADIENT_ACCUMULATION_STEPS}")
        print(f"Effective batch: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size}")
        print(f"Learning rate:   {LEARNING_RATE}")
        print(f"Epochs:          {NUM_TRAIN_EPOCHS}")
        print(f"Early stopping:  patience={EARLY_STOPPING_PATIENCE}")
        print("=" * 70)
    
    # Load processor
    if is_main_process:
        print("\nLoading processor...")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=LANGUAGE,
        task=TASK
    )
    
    # Load model
    if is_main_process:
        print("Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Required for gradient checkpointing
    )
    
    # Align pad_token_id to suppress warnings
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Configure generation
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    
    # Load datasets - only on main process first, then broadcast
    if is_main_process:
        print("\nLoading datasets...")
    
    dataset_dir = Path(DATASET_DIR)
    
    # Load datasets (all processes need to do this for distributed training)
    # But we can make it faster by loading chunks in parallel
    train_dataset = load_dataset_from_chunks(dataset_dir, "train")
    val_dataset = load_dataset_from_chunks(dataset_dir, "val")
    
    # Prepare datasets (tokenize labels) - only do once, cache handles the rest
    if is_main_process:
        print("\nPreparing datasets...")
    
    train_dataset = prepare_dataset(train_dataset, processor)
    val_dataset = prepare_dataset(val_dataset, processor)
    
    # Limit eval samples for faster evaluation
    if MAX_EVAL_SAMPLES is not None and len(val_dataset) > MAX_EVAL_SAMPLES:
        if is_main_process:
            print(f"Limiting eval dataset from {len(val_dataset):,} to {MAX_EVAL_SAMPLES:,} samples")
        val_dataset = val_dataset.shuffle(seed=42).select(range(MAX_EVAL_SAMPLES))
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_label_length=448,  # Whisper's max token length
    )
    
    # Compute metrics function
    compute_metrics = create_compute_metrics(processor)
    
    # Calculate training steps
    total_train_samples = len(train_dataset)
    effective_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size
    steps_per_epoch = total_train_samples // effective_batch_size
    total_steps = steps_per_epoch * NUM_TRAIN_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    if is_main_process:
        print(f"\nTraining plan:")
        print(f"  Total samples:     {total_train_samples:,}")
        print(f"  Effective batch:   {effective_batch_size}")
        print(f"  Steps per epoch:   {steps_per_epoch:,}")
        print(f"  Total steps:       {total_steps:,}")
        print(f"  Warmup steps:      {warmup_steps:,}")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # Batch size
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # Learning rate schedule
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=warmup_steps,
        
        # Training duration
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        
        # Regularization
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        label_smoothing_factor=LABEL_SMOOTHING,
        
        # Precision - BF16 for H200
        bf16=True,
        fp16=False,
        bf16_full_eval=True,
        
        # Optimizer
        optim="adamw_torch_fused",  # Fused AdamW is faster on H200
        
        # Evaluation & Saving
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        eval_on_start=True,  # Run evaluation before training starts
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        
        # Generation for evaluation
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        generation_num_beams=GENERATION_NUM_BEAMS,
        
        # Logging
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to=["tensorboard"],
        disable_tqdm=False,  # Ensure progress bar is shown
        
        # DataLoader
        dataloader_num_workers=4,  # Per GPU
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        
        # Distributed training
        ddp_find_unused_parameters=False,
        
        # Gradient checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # Other
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
    )
    
    # Custom callback to print CER during evaluation (only on main process)
    from transformers import TrainerCallback
    
    class CERLoggingCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            # Only print on main process
            if state.is_world_process_zero and "eval_cer" in metrics:
                cer = metrics["eval_cer"]
                wer = metrics.get("eval_wer", "N/A")
                print(f"\n{'='*50}")
                print(f"Step {state.global_step}: CER = {cer:.2f}%, WER = {wer}%")
                print(f"{'='*50}\n")
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[early_stopping_callback, CERLoggingCallback()],
    )
    
    # Train
    if is_main_process:
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70 + "\n")
    
    # Check for existing checkpoints and resume
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(OUTPUT_DIR, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
            if is_main_process:
                print(f"Resuming from checkpoint: {last_checkpoint}")
    
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save final model
    if is_main_process:
        print("\nSaving final model...")
        trainer.save_model(f"{OUTPUT_DIR}/final")
        processor.save_pretrained(f"{OUTPUT_DIR}/final")
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Skip final evaluation here - use evaluate_whisper.py separately
        # This avoids NCCL timeout issues with distributed generation
        print("\nTraining complete! Run evaluation separately with:")
        print(f"  python evaluate_whisper.py --model-path {OUTPUT_DIR}/final")
        
        # Best eval metrics from log history (load_best_model_at_end uses CER)
        eval_logs = [x for x in trainer.state.log_history if "eval_cer" in x]
        best_metrics = min(eval_logs, key=lambda x: x["eval_cer"]) if eval_logs else {}
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best CER:        {best_metrics.get('eval_cer', 'N/A')}%")
        print(f"Best WER:        {best_metrics.get('eval_wer', 'N/A')}%")
        print(f"Model saved to:  {OUTPUT_DIR}/final")
        print("=" * 70)


if __name__ == "__main__":
    main()