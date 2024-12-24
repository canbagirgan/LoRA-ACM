import os
import argparse
from dotenv import load_dotenv
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from utils import str2bool, load_dataset_from_csv, save_model_to_local

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # LoRA-related parameters
    parser.add_argument("--lrank", default=2, type=int, help="Enter LoRA rank in the form of integer. Suggested numbers 2, 4, 8, 16, 32, 64")
    parser.add_argument("--lalpha", default=16, type=int, help="Enter LoRA Alpha parameter")
    # Model parameters
    parser.add_argument("--llm_model", default="Llama-3.2-3B-Instruct-bnb-4bit", type=str, help="Enter LLM model name withou unsloth")
    parser.add_argument("--max_seq_length", default=4096, type=int, help="Enter the maximum sequence length")
    parser.add_argument("--load_in_4bit", default=True, type=str2bool, help="Using 4bit quantization or not. If True, then 4bit quatization is applied.")
    # Training parameters 
    parser.add_argument("--epoch", default=1, type=int, help="Number of epoch in training. Enter 0 if you use steps")
    parser.add_argument("--steps", default=0, type=int, help="Number of steps in training. Enter 0 if you epoch")
    parser.add_argument("--num_procs", default=20, type=int, help="Number of processes supported by CPU.")
    parser.add_argument("--per_device_train_batch_size", default=4, type=int, help="Training batch size per device")
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="Learning rate for training")
    parser.add_argument("--prompt_template", default="prompt_templates/default.txt", type=str, help="Prompt tamplate of training")
    args = parser.parse_args()

    load_dotenv()

    # Loading the dataset
    dataset_csv_path = os.getenv('DATASET_PATH')
    dataset = load_dataset_from_csv(dataset_csv_path)
    
    # Checking if CUDA is available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        raise ValueError("Cuda is not available :/")


    # Model names and parameters
    llm_model = args.llm_model
    model_name = f"unsloth/{llm_model}"
    max_seq_length = args.max_seq_length
    load_in_4bit = args.load_in_4bit
    dtype = None

    # Model LoRA hyperparams
    lrank = args.lrank
    lalpha = args.lalpha
    
    # Training parameters
    epoch = args.epoch
    per_device_train_batch_size = args.per_device_train_batch_size
    learning_rate = args.learning_rate
    prompt_template_path = args.prompt_template
    num_procs = args.num_procs

    # Defining fine-tuning model name
    finetuned_model_name = f"cs588_{llm_model}_r{lrank}_la{lalpha}_epoch{epoch}_bs{per_device_train_batch_size}_lr{learning_rate}"

    # Loading pretrained model weights
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    model.to(device)

    # Creating LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=lrank,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
                         # "gate_proj", "up_proj", "down_proj"
        lora_alpha = lalpha,
        lora_dropout = 0,
        bias = "none", 
        use_gradient_checkpointing = "unsloth",
        random_state = 588,
        use_rslora = False,
        loftq_config = None, 
    )

    # Reading prompt template
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(data_rows):
        diffs = data_rows["diff"]
        commit_messages = data_rows["msg"]
        texts = []
        for diff, commit_message in zip(diffs, commit_messages):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            # print(prompt_template)
            text = prompt_template.format(
            CODE_DIFFS = diff,
            COMMIT_MESSAGE = commit_message
            )
            text = text + EOS_TOKEN
            texts.append(text)
        return {"text" : texts}

    # Mapping dataset values to our input template
    dataset = dataset.map(formatting_prompts_func, batched=True)

    output_dir = f"training_outputs/{finetuned_model_name}/"

    # Starting the training process
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = num_procs,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = per_device_train_batch_size,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = epoch,
            learning_rate = learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 588,
            output_dir = output_dir,
            report_to = "none",
            save_strategy = "steps",
            save_steps = 100, #10000 # Number of updates steps before two checkpoint saves if save_strategy="steps"
            save_total_limit = 10 # Limit the total amount of checkpoints. Deletes the older checkpoints in output_dir
        )
    )

    trainer_stats = trainer.train()

    save_model_to_local(model, tokenizer, output_dir)