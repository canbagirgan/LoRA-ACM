#!/bin/bash

# List of values for rank and alpha
values=(2 4 8 16 32 64)

# Loop over values
for value in "${values[@]}"; do
    echo "Running training with lrank=$value and lalpha=$value"

    python3 train.py \
        --lrank $value \
        --lalpha $value \
        --llm_model "Llama-3.2-3B-Instruct-bnb-4bit" \
        --max_seq_length 4096 \
        --load_in_4bit True \
        --epoch 1 \
        --steps 0 \
        --per_device_train_batch_size 16 \
        --learning_rate 2e-4

    echo "Finished training for lrank=$value and lalpha=$value"
done


##  Possibe 4bit Models
# "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
# "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
# "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
# "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
# "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
# "unsloth/Phi-3-medium-4k-instruct",
# "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

# "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
# "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
# "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

##  Possibe Qwen Models
# "unsloth/Qwen2.5-Coder-32B-Instruct",      # Qwen 2.5 Coder 2x faster
# "unsloth/Qwen2.5-Coder-7B",
# "unsloth/Qwen2.5-14B-Instruct",            # 14B fits in a 16GB card
# "unsloth/Qwen2.5-7B",
# "unsloth/Qwen2.5-72B-Instruct",            # 72B fits in a 48GB card
