# LLM-RAC

This repo provides a library for buidling retrieval augmented classifications (RAC) systems, pairing embedding models with LLMs for enhanced text classification accuracy.

There are a few simple steps to building a RAC system:

1. Collect data of paired text and labels
2. Choose an embedding model
3. [Optional] Finetune the embedding model on your data
- We provide a logic for training the model, deriving pairs of positive and negatives using the provided data. Text with the same class label will be provided as positive examples, and text with different class labels will be provided as negative examples.
4. Build an ANN index by embedding all of the data, additionally storing metadata for each vector, e.g. the text and label.

Here we could build a simple RAC system using just the embedding model and the ANN index. For a given query, we search for the top-K similar examples and perform a majority vote on the labels. This is a simple baseline, but we can do better.

5. We use the top-K similar pairs as in-context learning examples which we add to the prompt for an LLM, which is instructed to make a prediction on the query.

We provide a notebook with samples of how to use this library as well as two scripts to build the embedding index and evaluate the RAC system.