# data_loading.py

import os
import pandas as pd

def load_data(data_path, sep=None):
    """
    Load the dataset from the specified file path.

    Args:
        data_path (str): The path to the dataset file.
        sep (str, optional): The separator used in the CSV file. If not provided, it will be automatically detected. Defaults to None.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the file extension is not supported or if the specified separator is not found in the CSV file.
    """
    # Check if the file exists
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"The specified file does not exist: {data_path}")

    # Get the file extension
    _, file_extension = os.path.splitext(data_path)

    # Load the dataset based on the file extension
    if file_extension == '.csv':
        if sep is None:
            # Automatically detect the separator
            with open(data_path, 'r') as file:
                first_line = file.readline()
            sep = ',' if ',' in first_line else ';' if ';' in first_line else '\t'
        
        try:
            df = pd.read_csv(data_path, sep=sep)
        except pd.errors.EmptyDataError:
            raise ValueError(f"The specified separator '{sep}' was not found in the CSV file.")
    
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(data_path)
    elif file_extension == '.json':
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    return df