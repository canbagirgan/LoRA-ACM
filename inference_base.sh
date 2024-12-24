
#!/bin/bash

#List of values for LLM models
llm_models=(
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    # "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    # "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
)

# Loop over models
for llm_model in "${llm_models[@]}"; do
    echo "Running inference with llm_model=$llm_model"

    python3 inference_base.py \
        --llm_model "$llm_model" \
        --max_seq_length 4096 \
        --load_in_4bit True \
        --inference_batch_size 1 \

    echo "Finished inference for llm_model=$llm_model"
done


## For llm_model checkout training_outputs for the available training models
## To use fine-tuned model with the name cs588_Llama-3.2-3B-Instruct-bnb-4bit_r2_la2_epoch1_bs16_lr0.0002
## you should enter "Llama-3.2-3B-Instruct-bnb-4bit" as the llm_model


## For base models, enter one of the following
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

