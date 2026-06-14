import json
import pandas as pd
from datasets import Dataset, DatasetDict

class DatasetLoader:
    """Handles loading raw data and persona mappings."""
    
    def __init__(self, train_path: str, val_path: str, test_path: str, persona_path: str):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.persona_path = persona_path

    def load_persona_map(self) -> dict:
        with open(self.persona_path, 'r') as f:
            personas = json.load(f)
        return {u['persona_id']: u for u in personas}

    def load_train_dataset(self) -> DatasetDict:
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        
        return DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "val": Dataset.from_pandas(val_df)
        })
    
    def load_test_dataset(self) -> DatasetDict:
        test_df = pd.read_csv(self.test_path)

        return DatasetDict({
            "test": Dataset.from_pandas(test_df),
        })
        