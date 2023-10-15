"""Script for evaling different RAC methods on a given dataset.

We will compare using an off-the-shelf model and finetuned model as k-NN classifiers. We will also compare using these
models as examples retrievers for an LLM-based classification.
"""
import argparse
import os
from typing import List, NamedTuple

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from rac.ann import ANN
from rac.llm import OpenAIClient
from rac.model_utils import train_model
from rac.rac import LLMRAC, RAC, LLMRACData


class SizeInfo(NamedTuple):
    """Information about the size of an index."""

    max_index_size: int
    batch_size: int


class EvalResults(NamedTuple):
    """Results of an evaluation."""

    accuracy: float
    predictions: List[str]


def load_prompt(prompt_path: str) -> str:
    """Load prompt from a file."""
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


def eval_on_data(test_data: pd.DataFrame, rac: RAC) -> EvalResults:
    """Evaluate a RAC on a dataset."""
    correct = 0
    predictions = []
    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        prediction = rac.predict(row["text"])
        predictions.append(prediction)
        if prediction == row["label"]:
            correct += 1
    return EvalResults(accuracy=correct / len(test_data), predictions=predictions)


def eval_log_and_print_rac_results(test_data: pd.DataFrame, rac: RAC, column_name: str):
    """Evaluate a RAC on a dataset, log and print results."""
    results = eval_on_data(test_data, rac)
    test_data[column_name] = results.predictions
    print(f"{column_name} accuracy: {results.accuracy}")


def eval_model(
    model: SentenceTransformer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    size_info: SizeInfo,
    prompt_path: str,
    prefix: str,
    output_path: str,
):
    """Evaluate a model as a RAC and LLMRAC"""
    pretrained_index = ANN.build_index_from_data(train_df, model, size_info.max_index_size, size_info.batch_size)

    rac = RAC(model, pretrained_index)

    llm_client = OpenAIClient(os.environ["OPENAI_API_KEY"], model_name="gpt-3.5-turbo")
    llm_rac_data = LLMRACData(k=5, min_score=0.0, instructions=load_prompt(prompt_path))
    llm_rac = LLMRAC.from_rac(rac, llm_client, llm_rac_data)

    eval_log_and_print_rac_results(test_df, rac, f"{prefix}_rac_predictions")
    rac.save(os.path.join(output_path, f"{prefix}_rac"))
    eval_log_and_print_rac_results(test_df, llm_rac, f"{prefix}_llm_rac_predictions")
    llm_rac.save(os.path.join(output_path, f"{prefix}_llm_rac"))


def main_eval(model_name_or_path: str, data: str, prompt_path: str, size_info: SizeInfo, output_path: str):
    """Main function for evaluating RAC methods"""
    data_frame = pd.read_csv(data)
    train_df, test_df = train_test_split(data_frame, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    model = SentenceTransformer(model_name_or_path)
    eval_model(model, train_df, test_df, size_info, prompt_path, "pretrained", output_path)

    model = train_model(model, train_df, test_df)
    eval_model(model, train_df, test_df, size_info, prompt_path, "finetuned", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates different RAC methods")
    parser.add_argument("--data", type=str, help="Path to data")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="all-mpnet-base-v2",
        help="model name or path to existing to use for building index",
    )
    parser.add_argument("--max_index_size", type=int, help="max size of index to build")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size to use when encoding")
    parser.add_argument("--prompt_path", type=str, help="path to file containing prompt")
    parser.add_argument("--output_path", type=str, help="path to save index", default="outputs")
    args = parser.parse_args()
    main_eval(
        args.model_name_or_path,
        args.data,
        args.prompt_path,
        SizeInfo(args.max_index_size, args.batch_size),
        args.output_path,
    )
