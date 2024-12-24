import os
import glob
import argparse
from dotenv import load_dotenv
from unsloth import FastLanguageModel
import torch
from torch.utils.data import DataLoader
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import TextStreamer
from unsloth import is_bfloat16_supported
from utils import str2bool, load_dataset_from_csv, save_model_to_local, save_model_responses, generate_llm_response, response_parser

if __name__ == '__main__':
    print("================================================================================Inference Base is started.")
    load_dotenv()
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--llm_model", default="Llama-3.2-3B-Instruct-bnb-4bit", type=str, help="Enter LLM model name withou unsloth")
    parser.add_argument("--max_seq_length", default=4096, type=int, help="Enter the maximum sequence length")
    parser.add_argument("--load_in_4bit", default=True, type=str2bool, help="Using 4bit quantization or not. If True, then 4bit quatization is applied.")
    # Training parameters 
    parser.add_argument("--inference_batch_size", default=4, type=int, help="Inference batch size per device")
    parser.add_argument("--prompt_template", default="prompt_templates/default.txt", type=str, help="Prompt tamplate of training")
    args = parser.parse_args()

    load_dotenv()

    dataset_csv_path = os.getenv('TEST_DATASET_PATH')
    print("dataset_csv_path: ", dataset_csv_path)
    dataset = load_dataset_from_csv(dataset_csv_path)
    dataset_start_index = 0
    dataset_end_index = 10000
    dataset = dataset.select(list(range(dataset_start_index, dataset_end_index)))
    print("dataset: ", dataset)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        raise ValueError("Cuda is not available. Without cuda you can only drink tea.")

    # Loading the model
    dtype = None
    model_name = args.llm_model
    if "/" in model_name:
        model_name = model_name.replace("/", "_")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.llm_model,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = args.load_in_4bit,
    )
    model.to(device)
    tokenizer.padding_side = "left" # Setting padding side to the left
    FastLanguageModel.for_inference(model) # Enabling 2x faster inference

    # Reading prompt template
    prompt_template_path = args.prompt_template
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
            text = prompt_template.format(
            CODE_DIFFS = diff,
            COMMIT_MESSAGE = "" # Putting empty text for the commit message part since this is an inference
            )
            # text = text + EOS_TOKEN # Only used in training
            text = text.replace("<nl>", "\n")
            texts.append(text)
        return { "text" : texts, }

    
    dataset = dataset.map(formatting_prompts_func, batched=True) # Mapping dataset values to the input template.
    prompts = dataset["text"]

    
    # Creating DataLoader for batching
    inference_batch_size = args.inference_batch_size
    dataloader = DataLoader(prompts, batch_size=inference_batch_size, shuffle=False)

    print("Total number of iterations (batches):", len(dataloader))

    # Generating responses in batches
    responses = []
    for batch_no, batch in enumerate(dataloader):
        print(f"Batch {batch_no}: Number of items = {len(batch)}") # Optional: Printing batch details
        batch_response = generate_llm_response(model, tokenizer, device, batch)
        parsed_responses = response_parser(batch_response)
        save_model_responses(model_name=model_name, responses=parsed_responses, dataset_start_index=dataset_start_index)


    

    
    