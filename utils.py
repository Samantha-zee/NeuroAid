import pandas as pd
import os

def load_datasets(data_dir="data"):
    dataframes = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv') or file.endswith('.txt') or file.endswith('.tsv') or file.endswith('.json'):
                try:
                    path = os.path.join(root, file)
                    if file.endswith('.csv'):
                        df = pd.read_csv(path, encoding='utf-8', error_bad_lines=False)
                    elif file.endswith('.tsv'):
                        df = pd.read_csv(path, sep='\t', encoding='utf-8')
                    elif file.endswith('.json'):
                        df = pd.read_json(path)
                    elif file.endswith('.txt'):
                        df = pd.read_csv(path, delimiter=';', header=None, encoding='utf-8', error_bad_lines=False)
                    dataframes.append(df)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    return dataframes
