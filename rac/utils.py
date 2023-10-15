"""Utility functions for RAC"""
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from rac.ann import ANN
from rac.model_utils import train_model
from rac.rac import RAC


def build_rac(model_name_or_path: str, data: str, output_path: str, train: bool, max_index_size: int, batch_size: int):
    """Main function for creating RAC index"""
    model = SentenceTransformer(model_name_or_path)
    data_frame = pd.read_csv(data)
    if train:
        train_df, test_df = train_test_split(data_frame, test_size=0.2, random_state=42)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        model = train_model(model, train_df, test_df)
    index = ANN.build_index_from_data(data_frame, model, max_index_size, batch_size)
    rac = RAC(model, index)
    rac.save(output_path)
