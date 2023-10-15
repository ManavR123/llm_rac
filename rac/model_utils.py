"""Code for loading and training models"""
from typing import List, Optional

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, evaluation, losses
from torch.utils.data import DataLoader


def create_examples(data: pd.DataFrame, k: int = 5) -> List[InputExample]:
    """Create examples for training a SentenceTransformer model. Each example is a pair of sentences.

    Args:
        data: dataframe with columns "text" and "label"
        k: number of negative examples to create for each positive example

    Returns:
        list of examples
    """
    labels = data["label"].unique()
    examples = []
    for label in labels:
        same_label_df = data[data["label"] == label]
        for i in range(len(same_label_df)):
            for j in range(i + 1, len(same_label_df)):
                examples.append(
                    InputExample(texts=[same_label_df["text"].iloc[i], same_label_df["text"].iloc[j]], label=1.0)
                )

    # Create negative samples
    negative_samples = []
    attempts = 0
    max_attempts = len(data) * 10  # or some other number based on your dataset size
    while len(negative_samples) < k * len(examples) and attempts < max_attempts:
        pair = data.sample(2)
        if pair["label"].iloc[0] != pair["label"].iloc[1]:  # make sure they have different labels
            negative_samples.append(InputExample(texts=[pair["text"].iloc[0], pair["text"].iloc[1]], label=0.0))
        attempts += 1

    if attempts == max_attempts:
        print(
            "Warning: Reached maximum number of attempts to create negative samples. \
            The number of negative samples may be less than desired."
        )

    examples.extend(negative_samples)

    return examples


def train_model(
    model: SentenceTransformer,
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame] = None,
    batch_size: int = 32,
) -> SentenceTransformer:
    """Train a SentenceTransformer model on a classification dataset.

    We implementation a contrastive learning algorithm for training the model.
    Examples that have the same class label are considered positive examples and
    examples that have different class labels are considered negative examples.

    We will split off 10% of the data for validation.

    Args:
        model: SentenceTransformer model to train
        train_data: training data
        val_data: validation data (optional)
        batch_size: batch size for training
        use_gpu: whether to use GPU for training

    Returns:
        trained SentenceTransformer model
    """
    train_dataloader = DataLoader(create_examples(train_data), shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    evaluator = None
    if val_data is not None:
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(create_examples(val_data))

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        evaluator=evaluator,
    )
    return model
