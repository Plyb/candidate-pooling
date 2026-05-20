import string
from collections.abc import Callable
from typing import Protocol

from byutils import load_model
from byutils import load_tokenizer
from nnsight import LanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from candidate_pooling.types import McqaExample, TokenizedExample

_ANSWER_LETTERS = list(string.ascii_uppercase)


def load_nnsight_model(model_id: str, model_cls: type[PreTrainedModel]) -> LanguageModel:
    hf_model = load_model(model_id, model_class=model_cls).cuda()
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(model_id)  # type: ignore[assignment]
    tokenizer.pad_token = tokenizer.eos_token
    return LanguageModel(hf_model, tokenizer=tokenizer)  # type: ignore[arg-type]


class PromptFormatter(Protocol):
    def format_prompt(self, example: McqaExample) -> str: ...

def _format_with_label(example: McqaExample, format_label: Callable[[int], str]) -> str:
    choices_str = "\n".join(
        f"{format_label(i)} {choice}"
        for i, choice in enumerate(example["choices"])
    )
    return f"Question: {example['question']}\n{choices_str}\nAnswer: ("

class DefaultPromptFormatter(PromptFormatter):
    def format_prompt(self, example: McqaExample) -> str:
        return _format_with_label(example, lambda i: f"{_ANSWER_LETTERS[i]})")
    

class PreMarkFormatter(PromptFormatter):
    def format_prompt(self, example: McqaExample) -> str:
        return _format_with_label(example, lambda i: f"- ({_ANSWER_LETTERS[i]})")


class PreMarkCorrectFormatter(PromptFormatter):
    def format_prompt(self, example: McqaExample) -> str:
        return _format_with_label(example, lambda i: f"{self._correct_mark(example, i)} ({_ANSWER_LETTERS[i]})")
    
    def _correct_mark(self, example: McqaExample, label_index: int) -> str:
        return '[x]' if example["answer"] == label_index else '[ ]'


class PreMarkIncorrectFormatter(PromptFormatter):
    def format_prompt(self, example: McqaExample) -> str:
        return _format_with_label(example, lambda i: f"{self._incorrect_mark(example, i)} ({_ANSWER_LETTERS[i]})")
    
    def _incorrect_mark(self, example: McqaExample, label_index: int) -> str:
        return '[x]' if example["answer"] == (label_index + 1 % len(example["choices"])) else '[ ]'



def make_tokenize_fn(model: LanguageModel, prompt_formatter: PromptFormatter = DefaultPromptFormatter()) -> Callable[[McqaExample, int], TokenizedExample]:
    tokenizer: PreTrainedTokenizerBase = model.tokenizer  # type: ignore[assignment]
    answer_ids: list[int] = tokenizer.convert_tokens_to_ids(_ANSWER_LETTERS)  # type: ignore[assignment]

    def tokenize(example: McqaExample, index: int) -> TokenizedExample:
        prompt = prompt_formatter.format_prompt(example)
        enc = tokenizer(prompt, return_tensors="pt")  # type: ignore[operator]
        return TokenizedExample(
            input_ids=enc["input_ids"][0],  # type: ignore[index]
            attention_mask=enc["attention_mask"][0],  # type: ignore[index]
            label_id=answer_ids[example["answer"]],
            example_id=index,
        )

    return tokenize
