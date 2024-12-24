import os
import glob
import argparse
from dotenv import load_dotenv
from unsloth import FastLanguageModel
import torch
from torch.utils.data import DataLoader
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from utils import str2bool, load_dataset_from_csv, save_model_to_local, save_model_responses, generate_llm_response, response_parser

if __name__ == '__main__':
    load_dotenv()
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
    parser.add_argument("--inference_batch_size", default=4, type=int, help="Inference batch size per device")
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="Learning rate for training")
    parser.add_argument("--prompt_template", default="prompt_templates/default.txt", type=str, help="Prompt tamplate of training")
    args = parser.parse_args()

    load_dotenv()

    dataset_csv_path = os.getenv('TEST_DATASET_PATH')
    dataset = load_dataset_from_csv(dataset_csv_path)
    dataset_start_index = 0
    dataset_end_index = 10000 
    dataset = dataset.select(list(range(dataset_start_index, dataset_end_index)))
    print("dataset: ", dataset)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        raise ValueError("Cuda is not available :/")


    # Model names and parameters
    llm_model = args.llm_model
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

    # Defining fine-tuning model directory
    finetuned_model = f"cs588_{llm_model}_r{lrank}_la{lalpha}_epoch{epoch}_bs{per_device_train_batch_size}_lr{learning_rate}"
    finetuned_model_dir = f"training_outputs/{finetuned_model}"
    print("finetuned_model_dir: ", finetuned_model_dir)
    
    # Checking if the directory exists
    if not os.path.exists(finetuned_model_dir):
        raise FileNotFoundError(f"Error: Fine-tuned model directory '{finetuned_model_dir}' does not exist.")

    # Checking for 'final_model' directory first
    final_model_path = os.path.join(finetuned_model_dir, "final_model")
    if os.path.exists(final_model_path):
        finetuned_model_name = final_model_path
        print(f"-------------------- Using final model: {finetuned_model_name}--------------------\n")
    else:
        # Searching for the latest checkpoint if 'final_model' is missing
        checkpoint_pattern = os.path.join(finetuned_model_dir, "checkpoint-*")
        checkpoints = glob.glob(checkpoint_pattern)

        if len(checkpoints) == 0:
            raise FileNotFoundError(f"Error: No checkpoints found in '{finetuned_model_dir}' and 'final_model' does not exist.")

        # Sorting checkpoints numerically to find the latest one
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        latest_checkpoint = checkpoints[-1]
        finetuned_model_name = latest_checkpoint
        print(f"-------------------- Using latest checkpoint: {finetuned_model_name}--------------------\n")

    # Loading pretrained model weights
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = finetuned_model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    model.to(device)
    # Model inference with Unsloth
    FastLanguageModel.for_inference(model)


    # Reading prompt template
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()

    EOS_TOKEN = tokenizer.eos_token 
    def formatting_prompts_func(data_rows):
        diffs = data_rows["diff"]
        commit_messages = data_rows["msg"]
        texts = []
        for diff, commit_message in zip(diffs, commit_messages):
            # If the length of the diff is larger than 128000, truncating the diff such that its lenght becomes 128000
            if len(diff) > 128000:
                diff = diff[:128000]
            # print(prompt_template)
            text = prompt_template.format(
            CODE_DIFFS = diff,
            COMMIT_MESSAGE = "" # Putting empty text for the commit message part since this is an inference
            )
            # text = text + EOS_TOKEN # Only used in training
            text = text.replace("<nl>", "\n")
            texts.append(text)
        return {"text" : texts}

    dataset = dataset.map(formatting_prompts_func, batched=True) # Mapping dataset values to the input template.
    prompts = dataset["text"]

    # Creating DataLoader for batching
    inference_batch_size = args.inference_batch_size
    dataloader = DataLoader(prompts, batch_size=inference_batch_size, shuffle=False)


    # Generating responses in batches
    responses = []
    for batch_no, batch in enumerate(dataloader):
        print(f"Batch {batch_no}: Number of items = {len(batch)}") # Optional: Printing batch details
        batch_response = generate_llm_response(model, tokenizer, device, batch)
        parsed_responses = response_parser(batch_response)
        save_model_responses(model_name=finetuned_model, responses=parsed_responses)