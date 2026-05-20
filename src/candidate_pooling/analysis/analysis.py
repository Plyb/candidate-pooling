import string
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.figure import Figure
from nnsight import LanguageModel
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from candidate_pooling.model import DefaultPromptFormatter, PromptFormatter
from candidate_pooling.types import McqaExample, TokenizedExample, to_transformer_input

_ANSWER_LETTERS = list(string.ascii_uppercase)


class SequenceTargeter(Protocol):
    def get_indices(self, example: McqaExample) -> list[int]: ...


class OptionPreTargeter(SequenceTargeter):
    def get_indices(self, example: McqaExample) -> list[int]:
        indices: list[int] = []
        cursor = len(f"Question: {example['question']}")
        indices.append(cursor)
        for i, choice in enumerate(example["choices"][:-1]):
            cursor += len(f"\n{_ANSWER_LETTERS[i]}) {choice}")
            indices.append(cursor)
        return indices


def _get_token_indices(
    example: McqaExample,
    targeter: SequenceTargeter,
    tokenizer: PreTrainedTokenizerBase,
    prompt_formatter: PromptFormatter = DefaultPromptFormatter(),
) -> Sequence[int]:
    prompt = prompt_formatter.format_prompt(example)
    enc = tokenizer(prompt, return_offsets_mapping=True)  # type: ignore[operator]
    offsets: list[tuple[int, int]] = enc["offset_mapping"]  # type: ignore[assignment,index]

    token_indices: list[int] = []
    for char_offset in targeter.get_indices(example):
        for tok_idx, (start, end) in enumerate(offsets):
            if start <= char_offset < end:
                token_indices.append(tok_idx)
                break
        else:
            raise ValueError(f"could not find token containing char offset {char_offset}")
    return token_indices


def _get_token_indices_by_example(
    tokenized_examples: Iterable[TokenizedExample],
    targetter: SequenceTargeter,
    tokenizer: PreTrainedTokenizerBase,
) -> Mapping[int, Sequence[int]]:
    tokenized_list = list(tokenized_examples)
    mcqa_examples = _load_mcqa_examples_for_tokenized(tokenized_list, tokenizer)
    return {
        tok_ex["example_id"]: _get_token_indices(mcqa_ex, targetter, tokenizer)
        for tok_ex, mcqa_ex in zip(tokenized_list, mcqa_examples)
    }


def _load_mcqa_examples_for_tokenized(
    tokenized_examples: Iterable[TokenizedExample],
    tokenizer: PreTrainedTokenizerBase,
) -> Iterable[McqaExample]:
    answer_ids: list[int] = tokenizer.convert_tokens_to_ids(_ANSWER_LETTERS)  # type: ignore[assignment]
    pos_by_id = {aid: i for i, aid in enumerate(answer_ids)}
    for ex in tokenized_examples:
        prompt: str = tokenizer.decode(ex["input_ids"], skip_special_tokens=True)  # type: ignore[assignment]
        body = prompt[: prompt.rindex("\nAnswer: (")]
        question_line, *choice_lines = body.split("\n")
        question = question_line[len("Question: "):]
        choices = [line[len("X) "):] for line in choice_lines]
        yield McqaExample(question=question, choices=choices, answer=pos_by_id[ex["label_id"]])


def _get_activations(
    tokenized_examples: Iterable[TokenizedExample],
    indices: Mapping[int, Sequence[int]],
    model: LanguageModel,
    layer: int = 8,
) -> Mapping[int, Float[Tensor, "num_options d_model"]]:
    result: dict[int, Float[Tensor, "num_options d_model"]] = {}
    for ex in tqdm(tokenized_examples):
        ex_id = ex["example_id"]
        ex_indices = list(indices[ex_id])
        with torch.no_grad(), model.trace(to_transformer_input(ex)):
            hidden = model.model.layers[layer].output[0].save()  # type: ignore[attr-defined]
        idx_tensor = torch.as_tensor(ex_indices, device=hidden.device)
        result[ex_id] = hidden[idx_tensor].detach()
    return result


@dataclass
class DiffResults:
    avg_diff: float
    avg_diff_by_answer_position: Sequence[float]
    stddev_diff: float
    stddev_diff_by_answer_position: Sequence[float]


@dataclass
class CategorizedActivations:
    by_pos_correct: Mapping[int, list[Tensor]]
    by_pos_incorrect: Mapping[int, list[Tensor]]


def _get_categorized_activations(
        tokenized: Iterable[TokenizedExample],
        pos_by_id: Mapping[int, int],
        activations_by_id: Mapping[int, Float[Tensor, "num_options d_model"]],
        basis_dir: Float[Tensor, "d_model"]
):
    v = basis_dir.cuda().float()

    by_pos_correct: dict[int, list[Tensor]] = {}
    by_pos_incorrect: dict[int, list[Tensor]] = {}

    for ex in tokenized:
        ex_id = ex["example_id"]
        correct_pos = pos_by_id[ex["label_id"]]
        projs = activations_by_id[ex_id].cuda().float() @ v
        for pos, proj in enumerate(projs):
            bucket = by_pos_correct if pos == correct_pos else by_pos_incorrect
            bucket.setdefault(pos, []).append(proj)

    return CategorizedActivations(
        by_pos_correct,
        by_pos_incorrect
    )


def _get_diff_averages(categorized_activations: CategorizedActivations) -> DiffResults:
    by_pos_correct = categorized_activations.by_pos_correct
    by_pos_incorrect = categorized_activations.by_pos_incorrect

    n_positions = max(
        max(by_pos_correct.keys(), default=-1),
        max(by_pos_incorrect.keys(), default=-1),
    ) + 1
    avg_diff_by_pos: list[float] = []
    stddev_diff_by_pos: list[float] = []
    valid_diffs: list[Tensor] = []
    valid_variances: list[Tensor] = []
    for pos in range(n_positions):
        c = by_pos_correct.get(pos, [])
        ic = by_pos_incorrect.get(pos, [])
        if len(c) >= 2 and len(ic) >= 2:
            ct = torch.stack(c)
            ict = torch.stack(ic)
            diff = ct.mean() - ict.mean()
            variance = ct.var(unbiased=True) / len(c) + ict.var(unbiased=True) / len(ic)
            avg_diff_by_pos.append(float(diff))
            stddev_diff_by_pos.append(float(variance.sqrt()))
            valid_diffs.append(diff)
            valid_variances.append(variance)
        else:
            avg_diff_by_pos.append(float("nan"))
            stddev_diff_by_pos.append(float("nan"))

    if valid_diffs:
        avg_diff = float(torch.stack(valid_diffs).mean())
        stddev_diff = float(torch.stack(valid_variances).sum().sqrt() / len(valid_variances))
    else:
        avg_diff = float("nan")
        stddev_diff = float("nan")

    return DiffResults(
        avg_diff=avg_diff,
        avg_diff_by_answer_position=avg_diff_by_pos,
        stddev_diff=stddev_diff,
        stddev_diff_by_answer_position=stddev_diff_by_pos,
    )

 

def _categorize_option_projections(
    tokenized_examples: Iterable[TokenizedExample],
    targeter: SequenceTargeter,
    basis_dir: Float[Tensor, "d_model"],
    model: LanguageModel,
    layer: int,
) -> CategorizedActivations:
    tokenizer: PreTrainedTokenizerBase = model.tokenizer  # type: ignore[assignment]
    answer_ids: list[int] = tokenizer.convert_tokens_to_ids(_ANSWER_LETTERS)  # type: ignore[assignment]
    pos_by_id = {aid: i for i, aid in enumerate(answer_ids)}

    tokenized_list = list(tokenized_examples)
    indices_by_id = _get_token_indices_by_example(tokenized_list, targeter, tokenizer)
    activations_by_id = _get_activations(tokenized_list, indices_by_id, model, layer)
    return _get_categorized_activations(tokenized_list, pos_by_id, activations_by_id, basis_dir)


def get_option_diff(
    tokenized_examples: Iterable[TokenizedExample],
    targeter: SequenceTargeter,
    basis_dir: Float[Tensor, "d_model"],
    model: LanguageModel,
    layer: int = 8,
) -> DiffResults:
    return _get_diff_averages(
        _categorize_option_projections(tokenized_examples, targeter, basis_dir, model, layer)
    )


def plot_option_diff(
    tokenized_examples: Iterable[TokenizedExample],
    targeter: SequenceTargeter,
    basis_dir: Float[Tensor, "d_model"],
    model: LanguageModel,
    layer: int = 8,
) -> Figure:
    return _plot_categorized(
        _categorize_option_projections(tokenized_examples, targeter, basis_dir, model, layer)
    )


def _plot_categorized(categorized: CategorizedActivations) -> Figure:
    by_pos_correct = categorized.by_pos_correct
    by_pos_incorrect = categorized.by_pos_incorrect
    n_positions = max(
        max(by_pos_correct.keys(), default=-1),
        max(by_pos_incorrect.keys(), default=-1),
    ) + 1

    def stack_to_numpy(samples: list[Tensor]) -> np.ndarray:
        if not samples:
            return np.array([])
        return torch.stack(samples).detach().float().cpu().numpy()

    correct_per_pos = [stack_to_numpy(by_pos_correct.get(p, [])) for p in range(n_positions)]
    incorrect_per_pos = [stack_to_numpy(by_pos_incorrect.get(p, [])) for p in range(n_positions)]

    correct_data = correct_per_pos + [np.concatenate(correct_per_pos) if any(len(a) for a in correct_per_pos) else np.array([])]
    incorrect_data = incorrect_per_pos + [np.concatenate(incorrect_per_pos) if any(len(a) for a in incorrect_per_pos) else np.array([])]
    labels = [_ANSWER_LETTERS[p] for p in range(n_positions)] + ["overall"]

    positions = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 1.2 + 2.0), 4.5))
    bp_c = ax.boxplot(correct_data, positions=positions - width / 2, widths=width, patch_artist=True, manage_ticks=False)
    bp_i = ax.boxplot(incorrect_data, positions=positions + width / 2, widths=width, patch_artist=True, manage_ticks=False)

    for box in bp_c["boxes"]:
        box.set_facecolor("lightblue")
    for box in bp_i["boxes"]:
        box.set_facecolor("lightsalmon")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("answer position")
    ax.set_ylabel("projection onto basis_dir")
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.legend([bp_c["boxes"][0], bp_i["boxes"][0]], ["correct", "incorrect"])
    fig.tight_layout()
    return fig