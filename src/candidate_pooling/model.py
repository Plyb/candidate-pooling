from typing import Iterator

import torch
from braided import strand
from braided.strand import OneToOne
from byutils import load_model
from byutils import load_tokenizer
from nnsight import LanguageModel
from transformers import PreTrainedTokenizerBase

from candidate_pooling.types import MmluExample, TokenizedExample

_ANSWER_LETTERS = ["A", "B", "C", "D"]
_CHOICE_PREFIXES = ["A) ", "B) ", "C) ", "D) "]


def load_nnsight_model(model_id: str) -> LanguageModel:
    hf_model = load_model(model_id).cuda()
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(model_id)  # type: ignore[assignment]
    tokenizer.pad_token = tokenizer.eos_token
    return LanguageModel(hf_model, tokenizer=tokenizer)  # type: ignore[arg-type]


def make_tokenize_strand(model: LanguageModel) -> OneToOne[TokenizedExample]:
    tokenizer: PreTrainedTokenizerBase = model.tokenizer  # type: ignore[assignment]
    answer_ids: list[int] = tokenizer.convert_tokens_to_ids(_ANSWER_LETTERS)  # type: ignore[assignment]

    @strand
    def tokenize(example: MmluExample) -> TokenizedExample:
        choices_str = "\n".join(
            f"{prefix}{choice}"
            for prefix, choice in zip(_CHOICE_PREFIXES, example["choices"])
        )
        prompt = f"Question: {example['question']}\n{choices_str}\nAnswer:"
        enc = tokenizer(prompt, return_tensors="pt")  # type: ignore[operator]
        return TokenizedExample(
            input_ids=enc["input_ids"][0],  # type: ignore[index]
            attention_mask=enc["attention_mask"][0],  # type: ignore[index]
            label_id=answer_ids[example["answer"]],
            example_id=example["example_id"],
        )

    return tokenize
