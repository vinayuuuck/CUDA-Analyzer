import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset


def create_prompt(row):
    """
    Create a text prompt for the LLM
    
    Input format:
    kernel_name: vectorAdd
    N: 4096
    exec_time: 0.052 ms
    occupancy: 0.85
    threads_per_block: 256
    dimensionality: 1D
    compute_intensity: 12.5
    has_shared_memory: False
    global_reads: 10
    global_writes: 5
    arithmetic_ops: 125
    memory_ops: 15
    
    Output format:
    block_dims: (256, 1, 1)
    total_threads: 256
    """
    
    prompt = f"""### Kernel: {row['kernel_name']}
    N: {int(row['N'])}
    dimensionality: {int(row['dimensionality'])}D
    compute_intensity: {row['compute_intensity']:.2f}
    has_shared_memory: {row['has_shared_memory']}
    global_reads: {int(row['global_reads'])}
    global_writes: {int(row['global_writes'])}
    arithmetic_ops: {int(row['arithmetic_ops'])}
    memory_ops: {int(row['memory_ops'])}

    ### Optimal Configuration:
    block_dims: ({int(row['block_x'])}, {int(row['block_y'])}, 1)
    total_threads: {int(row['block_x'] * row['block_y'])}
    """

    return prompt


def prepare_llm_dataset(csv_file: str, include_gpu=False):
    """
    Prepare dataset in LLM format
    
    Args:
        csv_file: Path to CSV with ALL configs (including suboptimal)
        include_gpu: If True, find optimal config per (kernel, N, GPU)
                    If False, find optimal config per (kernel, N) across all GPUs
    
    Returns:
        DataFrame with optimal configs and text prompts
    """
    df = pd.read_csv(csv_file)
    
    # Handle column name variations
    column_mapping = {
        'kernel': 'kernel_name',
        'bx': 'block_x',
        'by': 'block_y',
        'bz': 'block_z'
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Ensure required columns exist
    if 'block_x' not in df.columns:
        df['block_x'] = df['bx'] if 'bx' in df.columns else 0
    if 'block_y' not in df.columns:
        df['block_y'] = df['by'] if 'by' in df.columns else 0
    
    print(f"Loaded {len(df)} total configurations")
    print(f"Kernels: {df['kernel_name'].nunique()}")
    print(f"Data sizes: {sorted(df['N'].unique())}")
    
    # Group by kernel, N, and optionally GPU
    group_cols = ['kernel_name', 'N']
    if include_gpu and 'gpu' in df.columns:
        group_cols.append('gpu')
        print(f"GPUs: {df['gpu'].nunique()}")
    
    # Find optimal configs (min exec time for each group)
    print("\nFinding optimal configurations...")
    optimal_rows = []
    
    for group_key, group in df.groupby(group_cols):
        # Find row with minimum execution time
        best_idx = group['exec_time'].idxmin()
        best_row = group.loc[best_idx].copy()
        
        # Add some context: how much better is this than average?
        avg_time = group['exec_time'].mean()
        best_time = best_row['exec_time']
        speedup = avg_time / best_time
        
        best_row['speedup_vs_avg'] = speedup
        best_row['configs_tested'] = len(group)
        
        optimal_rows.append(best_row)
    
    df_optimal = pd.DataFrame(optimal_rows)
    
    print(f"✓ Found {len(df_optimal)} optimal configurations")
    print(f"  Configs per kernel: {len(df_optimal) / df['kernel_name'].nunique():.1f} avg")
    
    # Create prompts
    df_optimal['text'] = df_optimal.apply(create_prompt, axis=1)
    
    return df_optimal


def finetune_llm(df: pd.DataFrame, 
                 model_name: str = "gpt2",  # or "microsoft/phi-2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                 output_dir: str = "cuda_block_predictor_llm"):
    """
    Fine-tune LLM for block prediction
    
    Model options:
    - "gpt2" (124M params) - fast, good baseline
    - "gpt2-medium" (355M params) - better performance
    - "microsoft/phi-2" (2.7B params) - very capable, needs more GPU memory
    - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B params) - good balance
    """
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Prepare dataset
    dataset = Dataset.from_pandas(df[['text']])
    
    def tokenize_function(examples):
        # Tokenize texts
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split train/val
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
        eval_steps=100,
        save_steps=100,
        eval_strategy="steps",  # Changed from evaluation_strategy in newer transformers
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        push_to_hub=False,
        report_to="none"  # Disable wandb/tensorboard
    )
    
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ Model saved to: {output_dir}")
    
    return model, tokenizer


def predict_with_llm(model, tokenizer, kernel_info: dict):
    """
    Use fine-tuned LLM to predict block configuration
    
    Args:
        kernel_info: dict with kernel characteristics
        
    Returns:
        (block_x, block_y, total_threads)
    """
    # Create input prompt (without the output part)
    prompt = f"""### Kernel: {kernel_info['kernel_name']}
      N: {kernel_info['N']}
      dimensionality: {kernel_info['dimensionality']}D
      compute_intensity: {kernel_info['compute_intensity']:.2f}
      has_shared_memory: {kernel_info['has_shared_memory']}
      global_reads: {kernel_info['global_reads']}
      global_writes: {kernel_info['global_writes']}
      arithmetic_ops: {kernel_info['arithmetic_ops']}
      memory_ops: {kernel_info['memory_ops']}

      ### Optimal Configuration:
    """
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Get the device the model is on
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,  # Low temperature for more deterministic output
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract block dims from generated text
    # Look for pattern like "block_dims: (256, 1, 1)"
    import re
    
    match = re.search(r'block_dims:\s*\((\d+),\s*(\d+),\s*(\d+)\)', generated_text)
    if match:
        block_x = int(match.group(1))
        block_y = int(match.group(2))
        total_threads = block_x * block_y
        return block_x, block_y, total_threads
    else:
        # Fallback to default
        print("Warning: Could not parse LLM output, using default")
        return 256, 1, 256


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 finetune_llm.py <enriched_data.csv> [model_name]")
        print("\nModel options:")
        print("  - gpt2 (default, 124M params)")
        print("  - gpt2-medium (355M params)")
        print("  - microsoft/phi-2 (2.7B params, needs GPU)")
        print("  - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B params)")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
    
    print("CUDA Block Predictor - LLM Fine-tuning")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Data: {csv_file}")
    print()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ No GPU detected, training will be slow")
    
    # Prepare data
    print("\nPreparing dataset...")
    df = prepare_llm_dataset(csv_file)
    print(f"✓ Created {len(df)} training examples")
    
    # Show example
    print("\nExample training prompt:")
    print("-" * 60)
    print(df['text'].iloc[0])
    print("-" * 60)
    
    # Fine-tune
    model, tokenizer = finetune_llm(df, model_name=model_name)
    
    # Test prediction
    print("\n" + "=" * 60)
    print("TEST PREDICTION")
    print("=" * 60)
    
    test_kernel = {
        'kernel_name': 'matrixMul',
        'N': 2048,
        'dimensionality': 2,
        'compute_intensity': 15.2,
        'has_shared_memory': True,
        'global_reads': 20,
        'global_writes': 10,
        'arithmetic_ops': 304,
        'memory_ops': 30
    }
    
    bx, by, total = predict_with_llm(model, tokenizer, test_kernel)
    print(f"\nTest kernel: {test_kernel['kernel_name']}, N={test_kernel['N']}")
    print(f"Predicted: block=({bx}, {by}), total_threads={total}")


if __name__ == "__main__":
    main()