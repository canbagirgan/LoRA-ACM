import argparse
import os
import json
from datasets import load_dataset, DatasetDict, Dataset
from transformers import TextStreamer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Union, Dict
from huggingface_hub import login


def observe_dataset_distribution(dataset: Dataset, save_path: str = "dataset_distribution.png") -> Tuple[List[int], List[int]] :
    """
    The function plots the distribution of the dataset
    Arguments: 
    dataset (DatasetDict): Given dataset
    
    Returns
    bins (List[int]): List of integers that represent max word count for bins
    hist (List[int]): List of integers that represent number of commit messages in corresponding bin
    """
    # Getting the lengths of the messages in the dataset
    commit_messages = dataset['msg']
    message_lengths = [len(msg) for msg in commit_messages if msg]
    # Defining bin ranges (16 bins: 0-100, 100-200, ..., 1500-15000+)
    bins = list(range(100, 2000, 100)) # 0 to 2000
    hist, bin_edges = np.histogram(message_lengths, bins=bins)
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(hist)), hist, width=0.8,
            tick_label=[f"{bins[i]}-{bins[i+1]}" if i < len(bins) - 2 else "1500+" for i in range(len(hist))])
    plt.xlabel("Message Length Range")
    plt.ylabel("Number of Messages")
    plt.title("Distribution of Message Lengths")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Saving the plot
    plt.savefig(save_path)
    print(f"Plot saved at: {save_path}")
    plt.show()
    return bins, hist


def dataset_filter_acc_msg_len(dataset: Dataset, min_msg_len: int, max_msg_len: int) -> Dataset:
    """ 
    This function filters the dataset according to commit message length
    Arguments:
    dataset (Dataset): The dataset that is going to be filtered
    min_msg_len (int): Minimum message length
    max_msg_len (int): Maximum message length
    Returns:
    filtered_dataset (Dataset): A filtered Hugging Face Dataset where the length of the commit messages is between min_msg_len and max_msg_len.
    """
    # Filtering lambda function
    def filter_by_msg_length(row):
        # Handling None or empty 'msg' values
        if row['msg'] is None:
            return False
        msg_length = len(row['msg'])
        return min_msg_len <= msg_length < max_msg_len
    # Filtering dataset
    filtered_dataset = dataset.filter(filter_by_msg_length)
    
    return filtered_dataset

def str2bool(v: str) -> bool:
    """
    The function converst string boolean to boolean
    
    Arguments:
        v (str): string boolean
    
    Returns:
        Bool: corresponding boolean variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def load_dataset_from_csv(csv_file_path:str) -> Dataset:
    """
    The function loads dataset from csv file.

    Arguments:
    csv_file_path (str): path to the csv file

    Return:
    dataset (DatasetDict): HuggingFace datasetDict taken from csv file
    """
    dataset = load_dataset("csv", data_files=csv_file_path)['train']
    return dataset


def save_model_to_local(model, tokenizer, output_dir:str):
    """
    This function saves the model and tokenizer to local 

    Arguments:
    model: model itself
    tokenizer: the tokenizer of the model
    output_dir (str): save dir of the model
    """
    model.save_pretrained(f"{output_dir}/final_model") 
    tokenizer.save_pretrained(f"{output_dir}/final_model")



def save_model_responses(model_name: str, responses: List[str], dataset_start_index: int=0):
    """
    This function saves the model inference responses to a JSON file.
    If the file already exists, it appends the new responses.
    """
    # Defining the directory and file paths
    responses_dir_path = os.path.join(os.getcwd(), f"model_responses/{model_name}")
    
    # Ensuring the directory exists
    if not os.path.exists(responses_dir_path):
        os.makedirs(responses_dir_path)
    
    
    output_path = os.path.join(responses_dir_path, "model_response.json") # Defining the output file path

    # Loading existing responses if the file exists
    if os.path.exists(output_path):
        with open(output_path, "r") as json_file:
            existing_responses = json.load(json_file)
    else:
        existing_responses = {}

    # Assigning index as ID for the new responses, continuing from existing IDs
    start_index = len(existing_responses) + dataset_start_index
    new_responses_dict = {str(i + start_index): response for i, response in enumerate(responses)}

    # Merging existing and new responses
    existing_responses.update(new_responses_dict)

    # Writing the updated responses back to the JSON file
    with open(output_path, "w") as json_file:
        json.dump(existing_responses, json_file, indent=4)
    return



def generate_llm_response(model, tokenizer, device:str, prompts:List[str]) -> List[str]:
    """
    Generate responses from a language model for a list of prompts.

    Args:
        model: The language model for text generation.
        tokenizer: The tokenizer used for preparing input and decoding output.
        device (str): The device for running the model (e.g., "cuda" or "cpu").
        prompts (List[str]): A list of prompt strings.

    Returns:
        List[str]: The generated responses.
    """
    max_length = model.config.max_position_embeddings - 128 
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device) # tokenizing and moving input to the device.
    generated_ids = model.generate(
        **model_inputs, 
        # streamer = text_streamer, # Commenting out text streamer not to print outputs word by word
        max_new_tokens = 128)
    responses = tokenizer.batch_decode(generated_ids,  skip_special_tokens=True)
    return responses


def response_parser(response_list: List[str]):
    """ 
    This function parses LLM generated response such that it only takes the words after "Commit Message:"

    Arguments:
    response_list (List[str]): List of LLM responses

    Returns:
    parsed_responses (List[str]): List of parsed LLM responses
    """
    parsed_responses = []

    for response in response_list:
        # Splitting the response by "Commit Message:" and take the last part
        if "Commit Message for the given code diffs:" in response:
            parsed_responses.append(response.split("Commit Message for the given code diffs:")[-1].strip())
        else:
            # If "Commit Message:" is not present, take the response as the empty string
            parsed_responses.append("")
    return parsed_responses