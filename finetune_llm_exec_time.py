#!/usr/bin/env python3
"""
LLM Fine-tuning for CUDA Execution Time Prediction
Predicts exec_time given (kernel, N, block_x, block_y)

Uses ALL 15,962 samples to learn performance model.
At runtime: Try different block configs, ask LLM for predicted time, pick best.
"""

import pandas as pd
import torch
import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset


def create_exec_time_prompt(row):
    """
    Create prompt for execution time prediction

    Input: kernel characteristics + block config
    Output: predicted execution time
    """

    prompt = f"""### Kernel: {row['kernel_name']}
        N: {int(row['N'])}
        block_x: {int(row['block_x'])}
        block_y: {int(row['block_y'])}
        dimensionality: {int(row['dimensionality'])}D
        compute_intensity: {row['compute_intensity']:.2f}
        has_shared_memory: {row['has_shared_memory']}
        global_reads: {int(row['global_reads'])}
        global_writes: {int(row['global_writes'])}
        arithmetic_ops: {int(row['arithmetic_ops'])}
        memory_ops: {int(row['memory_ops'])}

        ### Execution Time:
        {row['exec_time']:.8f} ms
    """

    return prompt


def prepare_llm_dataset(csv_file: str):
    """
    Prepare ALL data for LLM training

    Key: Use ALL samples, not just optimal
    """
    df = pd.read_csv(csv_file)

    # Handle column naming
    column_mapping = {
        "kernel": "kernel_name",
        "bx": "block_x",
        "by": "block_y",
        "bz": "block_z",
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    print(f"Loaded {len(df):,} configurations")
    print(f"Kernels: {df['kernel_name'].nunique()}")
    print(f"Data sizes: {sorted(df['N'].unique())}")

    if "gpu" in df.columns:
        print(f"GPUs: {df['gpu'].nunique()}")

    # Create prompts for ALL samples
    print("\nCreating prompts for ALL samples...")
    df["text"] = df.apply(create_exec_time_prompt, axis=1)

    print(f"✓ Created {len(df):,} training examples")

    # Show statistics
    print(f"\nExec time statistics:")
    print(f"  Min: {df['exec_time'].min():.6f} ms")
    print(f"  Max: {df['exec_time'].max():.6f} ms")
    print(f"  Mean: {df['exec_time'].mean():.6f} ms")
    print(f"  Median: {df['exec_time'].median():.6f} ms")

    return df


def finetune_llm(
    df: pd.DataFrame, model_name: str = "gpt2", output_dir: str = "cuda_exec_time_llm"
):
    """
    Fine-tune LLM for execution time prediction
    """

    print(f"\nLoading model: {model_name}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Prepare dataset
    dataset = Dataset.from_pandas(df[["text"]])

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], truncation=True, max_length=512, padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    # Split
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    print(f"Training samples: {len(split_dataset['train']):,}")
    print(f"Validation samples: {len(split_dataset['test']):,}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # Fewer epochs for large dataset
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✓ Model saved to: {output_dir}")

    return model, tokenizer


def predict_exec_time_llm(model, tokenizer, kernel_info: dict):
    """
    Use LLM to predict execution time

    Args:
        kernel_info: Dict with kernel characteristics INCLUDING block config

    Returns:
        predicted_time in ms
    """
    # Create prompt (without the execution time part)
    prompt = f"""### Kernel: {kernel_info['kernel_name']}
N: {kernel_info['N']}
block_x: {kernel_info['block_x']}
block_y: {kernel_info['block_y']}
dimensionality: {kernel_info['dimensionality']}D
compute_intensity: {kernel_info['compute_intensity']:.2f}
has_shared_memory: {kernel_info['has_shared_memory']}
global_reads: {kernel_info['global_reads']}
global_writes: {kernel_info['global_writes']}
arithmetic_ops: {kernel_info['arithmetic_ops']}
memory_ops: {kernel_info['memory_ops']}

### Execution Time:
"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")

    # Get device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract time from generated text
    # Look for pattern like "0.012345 ms"
    match = re.search(r"([\d.]+)\s*ms", generated_text)
    if match:
        predicted_time = float(match.group(1))
        return predicted_time
    else:
        print(f"Warning: Could not parse LLM output: {generated_text[-100:]}")
        return 0.01  # default fallback


def find_optimal_config_llm(model, tokenizer, kernel_info, candidate_configs=None):
    """
    Find optimal block config using LLM as performance predictor

    This is the KEY function that replaces KLARAPTOR!
    """
    # Generate candidate configs if not provided
    if candidate_configs is None:
        dim = kernel_info.get("dimensionality", 1)

        if dim == 1:
            candidate_configs = [
                (32, 1),
                (64, 1),
                (128, 1),
                (256, 1),
                (512, 1),
                (1024, 1),
            ]
        elif dim == 2:
            candidate_configs = [
                (8, 4),
                (8, 8),
                (8, 16),
                (8, 32),
                (16, 4),
                (16, 8),
                (16, 16),
                (16, 32),
                (32, 4),
                (32, 8),
                (32, 16),
                (32, 32),
                (64, 4),
                (64, 8),
                (64, 16),
                (128, 4),
                (128, 8),
                (256, 4),
            ]
        else:
            candidate_configs = [(8, 8), (16, 8), (16, 16), (32, 8)]

    # Predict exec_time for each config
    predictions = []

    print(f"Testing {len(candidate_configs)} configurations...")

    for block_x, block_y in candidate_configs:
        # Create kernel info with this block config
        test_info = kernel_info.copy()
        test_info["block_x"] = block_x
        test_info["block_y"] = block_y

        # Predict
        pred_time = predict_exec_time_llm(model, tokenizer, test_info)

        predictions.append(
            {"block_x": block_x, "block_y": block_y, "predicted_time": pred_time}
        )

    # Find best
    best = min(predictions, key=lambda x: x["predicted_time"])

    return best["block_x"], best["block_y"], best["predicted_time"], predictions


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune LLM to predict CUDA execution time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This trains an LLM to predict exec_time given (kernel, N, block_x, block_y).
Uses ALL 15,962 samples to learn the performance model.

At runtime, try different block configs and pick the one with min predicted time.

Examples:
  # Train LLM
  python3 finetune_llm_exec_time.py klaraptor_enriched_data.csv gpt2-medium
  
  # Find optimal config
  python3 finetune_llm_exec_time.py --optimize \\
      --kernel Convolution2D_kernel --N 4096 --dim 2 --compute 8.3
        """,
    )

    parser.add_argument("csv_file", nargs="?", help="CSV with ALL configs")
    parser.add_argument(
        "model_name",
        nargs="?",
        default="gpt2",
        help="Model to fine-tune (gpt2, gpt2-medium, etc.)",
    )
    parser.add_argument("--output-dir", default="cuda_exec_time_llm")

    # Optimization mode
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--model-dir", default="cuda_exec_time_llm")
    parser.add_argument("--kernel", help="Kernel name")
    parser.add_argument("--N", type=int, help="Data size")
    parser.add_argument("--dim", type=int, help="Dimensionality")
    parser.add_argument("--compute", type=float, default=10.0)

    args = parser.parse_args()

    # Optimization mode
    if args.optimize:
        if not all([args.kernel, args.N, args.dim]):
            parser.error("Optimization requires: --kernel, --N, --dim")

        print("Loading fine-tuned LLM...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_dir)

        if torch.cuda.is_available():
            model = model.cuda()

        kernel_info = {
            "kernel_name": args.kernel,
            "N": args.N,
            "dimensionality": args.dim,
            "compute_intensity": args.compute,
            "has_shared_memory": 0,
            "global_reads": 10,
            "global_writes": 5,
            "arithmetic_ops": 100,
            "memory_ops": 15,
        }

        print(f"\nFinding optimal config for {args.kernel} with N={args.N}...")
        print()

        bx, by, pred_time, all_preds = find_optimal_config_llm(
            model, tokenizer, kernel_info
        )

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        print(f"\nOptimal configuration:")
        print(f"  block_dims: ({bx}, {by}, 1)")
        print(f"  predicted_time: {pred_time:.6f} ms")

        print(f"\nTop 10 configurations:")
        print("-" * 60)
        for pred in sorted(all_preds, key=lambda x: x["predicted_time"])[:10]:
            print(
                f"  ({pred['block_x']:3d}, {pred['block_y']:3d}): {pred['predicted_time']:.6f} ms"
            )

        return

    # Training mode
    if not args.csv_file:
        parser.error("CSV file required for training")

    print("=" * 60)
    print("CUDA EXECUTION TIME PREDICTOR (LLM)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.csv_file}")

    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ No GPU (training will be slow)")

    print()

    # Prepare data
    df = prepare_llm_dataset(args.csv_file)

    # Show example
    print("\nExample training prompt:")
    print("-" * 60)
    print(df["text"].iloc[0])
    print("-" * 60)

    # Fine-tune
    model, tokenizer = finetune_llm(
        df, model_name=args.model_name, output_dir=args.output_dir
    )

    # Test prediction
    print("\n" + "=" * 60)
    print("TEST PREDICTION")
    print("=" * 60)

    test_kernel = {
        "kernel_name": "Convolution2D_kernel",
        "N": 512,
        "block_x": 16,
        "block_y": 8,
        "dimensionality": 2,
        "compute_intensity": 8.3,
        "has_shared_memory": False,
        "global_reads": 9,
        "global_writes": 1,
        "arithmetic_ops": 83,
        "memory_ops": 10,
    }

    pred_time = predict_exec_time_llm(model, tokenizer, test_kernel)
    print(
        f"\nTest: {test_kernel['kernel_name']}, N={test_kernel['N']}, block=({test_kernel['block_x']},{test_kernel['block_y']})"
    )
    print(f"Predicted time: {pred_time:.6f} ms")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTo find optimal config:")
    print(f"  python3 {__file__} --optimize \\")
    print(f"    --kernel myKernel --N 4096 --dim 2")


if __name__ == "__main__":
    main()
