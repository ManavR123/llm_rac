"""RAC system."""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from typing import List

from sentence_transformers import SentenceTransformer

from rac.ann import ANN, ClassificationExample
from rac.llm import ChatMessage, ChatMessageRole, LLMClient, LLMCompletionRequest

RACDATA_FILE_NAME = "rac_data.json"


@dataclass
class RACData:
    """Properties of a RAC

    Args:
        k: number of nearest neighbors to retrieve
        min_score: minimum score for a result to be considered a match
    """

    k: int = 5
    min_score: float = 0.0


class RAC:
    """Retrieval Augmented Classification system."""

    def __init__(self, model: SentenceTransformer, index: ANN, rac_data: RACData = RACData()) -> None:
        self.model = model
        self.index = index
        self.rac_data = rac_data

    def save(self, output_path: str) -> None:
        """Save the RAC system to a directory."""
        os.makedirs(output_path, exist_ok=True)
        self.index.save(output_path)
        self.model.save(output_path)
        with open(os.path.join(output_path, RACDATA_FILE_NAME), "w", encoding="utf-8") as file:
            json.dump(asdict(self.rac_data), file)

    @classmethod
    def load(cls, output_path: str) -> RAC:
        """Load the RAC system from a directory."""
        index = ANN.load(output_path)
        model = SentenceTransformer(output_path)
        with open(os.path.join(output_path, RACDATA_FILE_NAME), "r", encoding="utf-8") as file:
            rac_data = RACData(**json.load(file))
        return cls(model, index, rac_data)

    def predict(self, text: str, debug: bool = False) -> str:
        """Predict the class of a query using k-NN and choosing the label by majority vote."""
        query_embedding = self.model.encode(text)
        results = self.index.get_similar_examples(query_embedding, self.rac_data.k, self.rac_data.min_score)
        if debug:
            logging.info("Results:\n%s", "\n".join([f"Text: {r.text} - Label: {r.label}" for r in results]))
        counter = Counter([result.label for result in results])
        return counter.most_common(1)[0][0]


@dataclass
class LLMRACData(RACData):
    """Properties of a LLMRAC

    Args:
        k: number of nearest neighbors to retrieve
        min_score: minimum score for a result to be considered a match
        instructions: instructions for the LLM. This should be a string that describes the task and only the task.
        You should be sure to define all of the possible labels for the task here as well.
    """

    instructions: str = ""

    def __post_init__(self) -> None:
        if self.instructions == "":
            raise ValueError("instructions must be provided and cannot be empty")


class LLMRAC(RAC):
    """LLM-driven RAC system."""

    def __init__(
        self,
        model: SentenceTransformer,
        index: ANN,
        client: LLMClient,
        llm_rac_data: LLMRACData,
    ) -> None:
        super().__init__(model, index)
        self.client = client
        self.rac_data = llm_rac_data

    def save(self, output_path: str) -> None:
        """Save the LLMRAC system to a directory."""
        super().save(output_path)
        self.client.save(output_path)

    @classmethod
    def load(cls, output_path: str) -> LLMRAC:
        """Load the LLMRAC system from a directory."""
        index = ANN.load(output_path)
        model = SentenceTransformer(output_path)
        client = LLMClient.from_path(output_path)
        with open(os.path.join(output_path, RACDATA_FILE_NAME), "r", encoding="utf-8") as file:
            llm_rac_data = LLMRACData(*json.load(file))
        return cls(model, index, client, llm_rac_data)

    @classmethod
    def from_rac(cls, rac: RAC, client: LLMClient, llm_rac_data: LLMRACData) -> LLMRAC:
        """Create an LLMRAC from a RAC."""
        return cls(rac.model, rac.index, client, llm_rac_data)

    def _format_prompt(self, text: str, results: List[ClassificationExample]) -> str:
        """Format a prompt for the LLM."""
        prompt = """Below you will be given instructions for a classification task. Your goal is to understand the task
        at hand and apply the instructions to classify the given text. You will also be given a set of examples that 
        are similar to the given text. You can use these examples to help you understand the task and the instructions.
        Be sure to output one and only one of the given labels as your answer.

        Instructions:
        """
        prompt += self.rac_data.instructions + "\n"
        prompt += "Now, here are some examples:\n"
        for result in results:
            prompt += f"Text: {result.text}\nResult: {result.label}\n"
        prompt += f"Text: {text}\nResult: "
        return prompt

    def predict(self, text: str, debug: bool = False) -> str:
        """Predict the class of a query using k-NN to select examples for the LLM which then predicts the label."""
        query_embedding = self.model.encode(text)
        results = self.index.get_similar_examples(query_embedding, self.rac_data.k, self.rac_data.min_score)
        prompt = self._format_prompt(text, results)
        if debug:
            logging.info("Prompt:\n%s", prompt)
        request = LLMCompletionRequest(
            messages=[
                ChatMessage(role=ChatMessageRole.USER, content=prompt),
            ]
        )
        resp = self.client.get_completion(request)
        if resp not in self.index.get_labels():
            raise ValueError(f"LLM returned invalid label: {resp}")
        return resp
