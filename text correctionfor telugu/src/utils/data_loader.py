# -*- coding: utf-8 -*-
import json
import pandas as pd
from datasets import load_dataset
import warnings

class DataLoader:
    """
    Utilities to load datasets for Telugu text correction.
    """
    
    @staticmethod
    def load_huggingface_dataset(dataset_name: str, config: str = None, split: str = "train"):
        """
        Loads datasets like IndicCorp, Aksharantar, or Dakshina from HF.
        """
        try:
            if config:
                dataset = load_dataset(dataset_name, config, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            return dataset
        except Exception as e:
            warnings.warn(f"Failed to load dataset {dataset_name}: {e}")
            return None
            
    @staticmethod
    def load_local_csv(filepath: str) -> pd.DataFrame:
        """
        Loads local CSV file.
        """
        try:
            return pd.read_csv(filepath, encoding='utf-8')
        except Exception as e:
            warnings.warn(f"Failed to load CSV from {filepath}: {e}")
            return pd.DataFrame()
            
    @staticmethod
    def load_local_json(filepath: str) -> list:
        """
        Loads local JSON file.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load JSON from {filepath}: {e}")
            return []
            
    @staticmethod
    def load_kaggle_dataset(path: str):
        """
        Stub for loading Kaggle dataset if downloaded locally.
        """
        return DataLoader.load_local_csv(path)
