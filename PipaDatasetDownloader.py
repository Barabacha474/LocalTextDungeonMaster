import pandas as pd
import os
from datasets import load_dataset
import json


def download_pippa(folder_path="pippa_data", file_name="pippa_chatml.jsonl"):
    """
    Downloads the Pippa dataset to the specified folder.

    Args:
        folder_path (str): Path to the folder where dataset will be saved
        file_name (str): Name of file that dataset will be stored in
    """

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    dataset_name = "chimbiwide/pippa"

    print(f"Downloading {dataset_name} dataset to '{folder_path}'...")

    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_name)

        # Convert to pandas DataFrame
        # Note: The dataset might have splits (train, validation, test)
        # We'll use the train split if available, otherwise the entire dataset
        if isinstance(dataset, dict):
            # If dataset has multiple splits, use train split
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                # Take the first split available
                first_split = list(dataset.keys())[0]
                df = dataset[first_split].to_pandas()
        else:
            # If it's a single Dataset object
            df = dataset.to_pandas()

        # Save to JSONL file in the specified folder
        output_path = os.path.join(folder_path, file_name)
        df.to_json(output_path, orient='records', lines=True)

        print("Dataset downloaded successfully!")
        print(f"Total rows: {len(df)}")
        print(f"File location: {os.path.join(os.getcwd(), folder_path)}")

        return df

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install pandas datasets")
        print("2. Logged into Hugging Face: huggingface-cli login")
        print("3. Have internet connection")
        return None

def download_pippa_filtered(folder_path="pippa_data", file_name="pippa_filtered_chatml.jsonl"):
    """
    Downloads the Pippa filtered dataset to the specified folder.

    Args:
        folder_path (str): Path to the folder where dataset will be saved
        file_name (str): Name of file that dataset will be stored in
    """

    dataset_name = "chimbiwide/pippa_filtered"

    print(f"Downloading {dataset_name} dataset to '{folder_path}'...")

    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_name)

        # Convert to pandas DataFrame
        # Note: The dataset might have splits (train, validation, test)
        # We'll use the train split if available, otherwise the entire dataset
        if isinstance(dataset, dict):
            # If dataset has multiple splits, use train split
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                # Take the first split available
                first_split = list(dataset.keys())[0]
                df = dataset[first_split].to_pandas()
        else:
            # If it's a single Dataset object
            df = dataset.to_pandas()

        # Save to JSONL file in the specified folder
        output_path = os.path.join(folder_path, file_name)
        df.to_json(output_path, orient='records', lines=True)

        print("Dataset downloaded successfully!")
        print(f"Total rows: {len(df)}")
        print(f"File location: {os.path.join(os.getcwd(), folder_path)}")

        return df

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install pandas datasets")
        print("2. Logged into Hugging Face: huggingface-cli login")
        print("3. Have internet connection")
        return None


if __name__ == "__main__":
    # Specify your folder path here (change if needed)
    folder_path = "Datasets/PIPPA"

    df = download_pippa(folder_path)
    print(df.head(1))
    df = download_pippa_filtered(folder_path)
    print(df.head(1))