"""Runner file for testing RAC library"""
import argparse

from rac.utils import build_rac

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds an index for enabling Retrieval Augmented Classification")
    parser.add_argument("--data", type=str, help="Path to data")
    parser.add_argument("--output_path", type=str, help="path to save index")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="all-mpnet-base-v2",
        help="model name or path to existing to use for building index",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="whether to train the model on the dataset before indexing",
    )
    parser.add_argument("--max_index_size", type=int, help="max size of index to build")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size to use when encoding")
    args = parser.parse_args()
    build_rac(
        args.model_name_or_path,
        args.data,
        args.output_path,
        args.train,
        args.max_index_size,
        args.batch_size,
    )
