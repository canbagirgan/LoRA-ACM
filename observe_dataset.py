import os
from dotenv import load_dotenv
from utils import load_dataset_from_csv, observe_dataset_distribution, dataset_filter_acc_msg_len



if __name__ == '__main__':
    load_dotenv() # loading variables into os.environ
    train_dataset_csv_path = os.getenv('TEST_DATASET_PATH')
    valid_dataset_csv_path = os.getenv('VALID_DATASET_PATH')
    test_dataset_csv_path = os.getenv('TEST_DATASET_PATH')

    dataset_distribution_path = "dataset_distributions/"
    # Checking if the dataset_distribution directory exists, and create it if not
    if not os.path.exists(dataset_distribution_path):
        os.makedirs(dataset_distribution_path)

    # Observing the train dataset
    train_dataset = load_dataset_from_csv(train_dataset_csv_path)
    bins, hist = observe_dataset_distribution(train_dataset, save_path="dataset_distributions/train_dataset_distribution.png")
    print("----- Train Dataset ------")
    print(f"bins: {bins}")
    print(f"hist: {hist}")

    # Filtering dataset according to commit message lenth
    train_filtered_dataset = dataset_filter_acc_msg_len(train_dataset, min_msg_len=0, max_msg_len=200)
    bins, hist = observe_dataset_distribution(train_filtered_dataset, save_path="dataset_distributions/train_dataset_distribution_filtered.png")
    print("----- Filtered Train Dataset ------")
    print(f"bins: {bins}")
    print(f"hist: {hist}")

    
    # Observing the test dataset
    test_dataset = load_dataset_from_csv(test_dataset_csv_path)
    bins, hist = observe_dataset_distribution(test_dataset, save_path="dataset_distributions/test_dataset_distribution.png")
    print("----- Test Dataset ------")
    print(f"bins: {bins}")
    print(f"hist: {hist}")

    # Filtering dataset according to commit message lenth
    test_filtered_dataset = dataset_filter_acc_msg_len(test_dataset, min_msg_len=0, max_msg_len=200)
    bins, hist = observe_dataset_distribution(test_filtered_dataset, save_path="dataset_distributions/test_dataset_distribution_filtered.png")
    print("----- Filtered Test Dataset ------")
    print(f"bins: {bins}")
    print(f"hist: {hist}")


    
