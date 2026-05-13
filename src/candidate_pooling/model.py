import string
from collections.abc import Callable

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


def make_tokenize_fn(model: LanguageModel) -> Callable[[McqaExample, int], TokenizedExample]:
    tokenizer: PreTrainedTokenizerBase = model.tokenizer  # type: ignore[assignment]
    answer_ids: list[int] = tokenizer.convert_tokens_to_ids(_ANSWER_LETTERS)  # type: ignore[assignment]

    def tokenize(example: McqaExample, index: int) -> TokenizedExample:
        choices_str = "\n".join(
            f"{_ANSWER_LETTERS[i]}) {choice}"
            for i, choice in enumerate(example["choices"])
        )
        prompt = f"Question: {example['question']}\n{choices_str}\nAnswer: ("
        enc = tokenizer(prompt, return_tensors="pt")  # type: ignore[operator]
        return TokenizedExample(
            input_ids=enc["input_ids"][0],  # type: ignore[index]
            attention_mask=enc["attention_mask"][0],  # type: ignore[index]
            label_id=answer_ids[example["answer"]],
            example_id=index,
        )

    return tokenize
