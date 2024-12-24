import sys
import json
import re
import csv
import argparse
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import corpus_bleu 
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction


csv.field_size_limit(sys.maxsize)

def extract_messages_from_csv(file_path, msize=0, num_rows=None):
    """
    Reads a CSV file, extracts messages based on custom logic,
    and returns them as a list.

    Args:
        file_path (str): Path to the CSV file.
        num_rows (int): Number of rows to process.

    Returns:
        list: List of extracted messages.
    """
    messages = defaultdict(str)
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            if num_rows:
                if i >= num_rows+1:
                    break

            line = ','.join(row)

            attributes = re.split(r',(?=\S)', line, maxsplit=2)
            
            if len(attributes[1]) > msize and len(attributes) != 0:
                messages[i] = attributes[1].strip()
            if msize == 0 and len(attributes) == 0:
                messages[i] = ""

    return messages

def extract_messages_from_json(file_path, num_rows=None):
    """
    Reads a JSON file, cleans and extracts meaningful messages.

    Args:
        file_path (str): Path to the JSON file.
        num_rows (int): Number of rows to process. If None, process all rows.

    Returns:
        list: A list of cleaned and extracted messages.
    """
    cleaned_messages = defaultdict(str)

    with open(file_path, mode='r', encoding='utf-8') as file:
        data = json.load(file)

    for i, raw_msg in enumerate(data.values()):
        if num_rows:
            if i >= num_rows:
                break
        
        msg = re.sub(r'```', '', raw_msg)
        
        msg = re.sub(r'\\[a-z0-9]{1,6}', '', msg)
        
        msg = re.sub(r'[^a-zA-Z0-9\s.,:;\'\"()\[\]\-_]', '', msg)
        msg = re.sub(r'\s+', ' ', msg).strip()

        cleaned_messages[i] = msg

    return cleaned_messages

def calculate_bleu4_score(references, predictions):
    """
    Calculates BLEU-4 score between two lists of sentences.

    Args:
        references (list of str): List of reference sentences (labels).
        predictions (list of str): List of model-predicted sentences.

    Returns:
        float: BLEU-4 score.
    """

    smoothing_function = SmoothingFunction().method1
    references_tokenized = [[ref.split()] for ref in references]
    predictions_tokenized = [pred.split() for pred in predictions]
    bleu4_score = nltk.translate.bleu_score.corpus_bleu(references_tokenized, predictions_tokenized, weights=(0.25, 0.25, 0.25, 0.25),
                                                        smoothing_function=smoothing_function)
    return bleu4_score

def calculate_rouge_score(references, predictions):
    """
    Calculates ROUGE-L score (precision, recall, F1) between two lists of sentences.

    Args:
        references (list of str): List of reference sentences (labels).
        predictions (list of str): List of model-predicted sentences.

    Returns:
        dict: Average ROUGE-L precision, recall, and F1 scores.
    """

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    total_precision, total_recall, total_f1 = 0, 0, 0
    n = len(references)

    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        total_precision += scores['rougeL'].precision
        total_recall += scores['rougeL'].recall
        total_f1 += scores['rougeL'].fmeasure

    avg_precision = total_precision / n
    avg_recall = total_recall / n
    avg_f1 = total_f1 / n

    return {
        "ROUGE-L Precision": avg_precision,
        "ROUGE-L Recall": avg_recall,
        "ROUGE-L F1": avg_f1
    }

def calculate_meteor_scores(references, predictions):
    """
    Calculates the average METEOR score between two lists of sentences.

    Args:
        references (list of str): List of reference sentences.
        predictions (list of str): List of predicted sentences.

    Returns:
        float: Average METEOR score.
    """
    scores = [
        meteor_score([ref.split()], cand.split()) 
        for ref, cand in zip(references, predictions)
    ]
    return sum(scores) / len(scores) 

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_path", type=str, help="Path of the file that contains labels")
    parser.add_argument("--result_path", type=str, help="Path of the file that contains model results")
    parser.add_argument("--outfile", type=str, help="Output file")
    parser.add_argument("--nrows", default=0, type=int, help="Number of label-result pair you want for eval (default is full range)")
    parser.add_argument("--msize", type=int, default=0, help="Min commit message lenght")
    args = parser.parse_args()

    return args

def main(args):
    csv_file_path = args.label_path 
    json_file_path = args.result_path
    nrows = None if args.nrows == 0 else args.nrows
    labels = extract_messages_from_csv(csv_file_path, msize=args.msize, num_rows=nrows)
    results = extract_messages_from_json(json_file_path, num_rows=nrows)

    labels_list, results_list = [],[]
    for i in labels.keys():
        labels_list.append(labels[i])
        results_list.append(results[i])

    assert len(results_list) == len(labels_list), "Length discrepancy."

    bleu4_score = calculate_bleu4_score(labels_list, results_list)
    print(f"BLEU-4 Score: {bleu4_score:.4f}")

    meteor_score = calculate_meteor_scores(labels_list, results_list)
    print(f"METEOR Score: {meteor_score:.4f}")

    rouge_scores = calculate_rouge_score(labels_list, results_list)
    print("-ROUGE-L Scores-")

    for metric, value in rouge_scores.items():
        print(f"{metric}: {value:.4f}")

    with open(args.outfile, "a") as fp:
        fp.write(f"BLEU-4 Score: {bleu4_score:.6f}\n")
        fp.write(f"METEOR Score: {meteor_score:.6f}\n")
        for metric, value in rouge_scores.items():
            fp.write(f"{metric}: {value:.6f}\n")

    return 0

if __name__=="__main__":
    exit(main(get_args()))
