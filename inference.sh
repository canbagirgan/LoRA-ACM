#!/bin/bash

# List of values for rank and alpha values
values=(32 16 8 4 2)


# Loop over values
for value in "${values[@]}"; do
    echo "Running inference with lrank=$value and lalpha=$value"

    python3 inference.py \
        --lrank $value \
        --lalpha $value \
        --llm_model "Llama-3.2-3B-Instruct-bnb-4bit" \
        --max_seq_length 4096 \
        --load_in_4bit True \
        --epoch 1 \
        --steps 0 \
        --per_device_train_batch_size 32 \
        --inference_batch_size 1 \
        --learning_rate 2e-4

    echo "Finished inference for lrank=$value and lalpha=$value"
done


## For llm_model checkout training_outputs for the available training models
## To use fine-tuned model with the name cs588_Llama-3.2-3B-Instruct-bnb-4bit_r2_la2_epoch1_bs16_lr0.0002
## you should enter "Llama-3.2-3B-Instruct-bnb-4bit" as the llm_model


# Model names
# cs588_Llama-3.2-3B-Instruct-bnb-4bit_r2_la2_epoch1_bs32_lr0.0002
# cs588_Llama-3.2-3B-Instruct-bnb-4bit_r4_la4_epoch1_bs32_lr0.0002
# cs588_Llama-3.2-3B-Instruct-bnb-4bit_r8_la8_epoch1_bs32_lr0.0002
# cs588_Llama-3.2-3B-Instruct-bnb-4bit_r16_la16_epoch1_bs32_lr0.0002
# cs588_Llama-3.2-3B-Instruct-bnb-4bit_r32_la32_epoch1_bs32_lr0.0002
