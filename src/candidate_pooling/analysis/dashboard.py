"""
there are three functions that create ipywidget dashboard components from a candidate pooling cached run

each takes in a cached run's path

each creates the widget with a dropdown that selects between the `BasisDirection`s saved in the cache

They create:
1. a widget showing the original example from the training set which the basis direction came from, with the token it came from highlighted and the correct answer shown
2. a widget showing the top k examples from the probe set by their loss fingerprint. Tokens are highlighted according to their loss fingerprints, with hover showing the exact loss fingerprint.
  - the examples are deduplicated, so if some of the top k loss fingerprints came from the same example, they are merged together and others fill in until k examples are shown
3. a widget showing histograms of the loss and entropy fingerprints

if possible, abstract out the dropdown logic

"""
import html
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from IPython.display import display
from nnsight import LanguageModel
from transformers import PreTrainedTokenizerBase

from byutils import load_tokenizer

from candidate_pooling.fingerprint import ALPHA_DEFAULT, compute_delta, make_baseline_fn
from candidate_pooling.lib.typed_dataset import TypedDataset
from candidate_pooling.mining import LAYER, compute_neg_gradients
from candidate_pooling.pipeline import MODEL_ID
from candidate_pooling.types import TokenizedExample


BasisRow = dict[str, Any]


def _load(path: Path) -> TypedDataset[dict[str, Any]]:
    return TypedDataset[dict[str, Any]](cast(Dataset, load_from_disk(str(path))))


def _load_basis(cache_path: Path) -> list[BasisRow]:
    return list(_load(cache_path / "out"))


def _basis_dropdown(
    basis_list: list[BasisRow],
    render: Callable[[BasisRow], widgets.Widget],
) -> widgets.Widget:
    options = [(f"Cluster {b['cluster_id']} (ex {b['example_id']})", i) for i, b in enumerate(basis_list)]
    dropdown = widgets.Dropdown(options=options, description="Basis:")
    container = widgets.VBox([])

    def show(idx: int) -> None:
        container.children = (render(basis_list[idx]),)

    dropdown.observe(lambda change: show(int(change["new"])), names="value")
    show(int(dropdown.value))  # type: ignore[arg-type]
    return widgets.VBox([dropdown, container])


def _decode_token(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    decoded = tokenizer.decode([token_id])
    return decoded if isinstance(decoded, str) else "".join(decoded)


def _visualize_invisibles(token: str) -> str:
    return token.replace("\n", "↵\n").replace("\t", "→\t").replace("\r", "␍")


_TOKEN_STYLE = "<style>.tok:hover { outline: 1px solid #000; }</style>"


def source_example_widget(cache_path: Path) -> widgets.Widget:
    basis_list = _load_basis(cache_path)
    train_by_ex = {row["example_id"]: row for row in _load(cache_path / "tok_train")}
    mined_by_ex: dict[int, list[dict[str, Any]]] = {}
    for cand in _load(cache_path / "mined"):
        mined_by_ex.setdefault(cand["example_id"], []).append(cand)
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    def find_token_pos(basis: BasisRow) -> int:
        target = np.asarray(basis["vector"], dtype=np.float32)
        for cand in mined_by_ex[basis["example_id"]]:
            if np.allclose(np.asarray(cand["vector"], dtype=np.float32), target, atol=1e-6):
                return int(cand["token_pos"])
        raise ValueError(f"no mined candidate matched basis for example {basis['example_id']}")

    def render(basis: BasisRow) -> widgets.Widget:
        ex = train_by_ex[basis["example_id"]]
        token_pos = find_token_pos(basis)
        spans = []
        for i, tid in enumerate(ex["input_ids"]):
            tok = html.escape(_visualize_invisibles(_decode_token(tokenizer, int(tid))))
            style = "background:#ffeb3b;font-weight:bold" if i == token_pos else ""
            spans.append(f"<span class='tok' style='{style}'>{tok}</span>")
        answer = html.escape(_decode_token(tokenizer, int(ex["label_id"])))
        return widgets.HTML(
            f"{_TOKEN_STYLE}"
            f"<pre style='white-space:pre-wrap;font-family:monospace'>{''.join(spans)}</pre>"
            f"<p><b>Correct answer:</b> {answer}</p>"
        )

    return _basis_dropdown(basis_list, render)


def top_probe_examples_widget(cache_path: Path, k: int = 10) -> widgets.Widget:
    basis_list = _load_basis(cache_path)
    probe_list = list(_load(cache_path / "tok_probe"))
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    lengths = [len(ex["input_ids"]) for ex in probe_list]
    bounds = np.cumsum([0, *lengths])

    def split_by_example(fingerprint: list[float]) -> list[np.ndarray]:
        arr = np.asarray(fingerprint, dtype=np.float32)
        return [arr[bounds[i] : bounds[i + 1]] for i in range(len(probe_list))]

    def render(basis: BasisRow) -> widgets.Widget:
        per_ex = split_by_example(basis["loss_fingerprint"])
        min_delta = np.array([float(p.min()) for p in per_ex])
        top_idx = np.argsort(min_delta)[:k]
        parts: list[str] = []
        for rank, idx in enumerate(top_idx):
            ex = probe_list[int(idx)]
            deltas = per_ex[int(idx)]
            scale = max(float(np.abs(deltas).max()), 1e-12)
            spans = []
            for i, tid in enumerate(ex["input_ids"]):
                tok = html.escape(_visualize_invisibles(_decode_token(tokenizer, int(tid))))
                d = float(deltas[i])
                alpha = min(abs(d) / scale, 1.0)
                rgb = "255,80,80" if d > 0 else "80,80,255"
                spans.append(
                    f"<span class='tok' style='background:rgba({rgb},{alpha:.2f})' "
                    f"title='Δloss={d:+.4f}'>{tok}</span>"
                )
            answer = html.escape(_decode_token(tokenizer, int(ex["label_id"])).strip())
            parts.append(
                f"<div style='margin-bottom:0.5em'>"
                f"<b>#{rank + 1}</b> example {ex['example_id']} &middot; "
                f"min Δloss={min_delta[idx]:+.4f} &middot; correct: <b>{answer}</b>"
                f"<pre style='white-space:pre-wrap;font-family:monospace;margin:0.25em 0'>"
                f"{''.join(spans)}</pre></div>"
            )
        return widgets.HTML(_TOKEN_STYLE + "<hr>".join(parts))

    return _basis_dropdown(basis_list, render)


def fingerprint_histograms_widget(cache_path: Path) -> widgets.Widget:
    basis_list = _load_basis(cache_path)
    probe_list = list(_load(cache_path / "tok_probe"))
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    per_token_answer: list[str] = []
    for ex in probe_list:
        letter = _decode_token(tokenizer, int(ex["label_id"])).strip()
        per_token_answer.extend([letter] * len(ex["input_ids"]))
    answer_arr = np.asarray(per_token_answer)
    answer_labels = sorted(set(per_token_answer))

    def render(basis: BasisRow) -> widgets.Widget:
        out = widgets.Output()
        with out:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            for ax, key, label in [
                (axs[0], "loss_fingerprint", "Δloss"),
                (axs[1], "entropy_fingerprint", "Δentropy"),
            ]:
                arr = np.asarray(basis[key], dtype=np.float32)
                mask = arr != 0
                nonzero = arr[mask]
                nonzero_density = len(nonzero) / max(len(arr), 1)
                groups = [nonzero[answer_arr[mask] == a] for a in answer_labels]
                ax.hist(groups, bins=50, stacked=True, label=answer_labels)
                ax.set_title(f"{label} (non-zero density: {nonzero_density:.2%})")
                ax.set_xlabel(label)
                ax.legend(title="correct", fontsize=8)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
        return out

    return _basis_dropdown(basis_list, render)


def _to_tokenized(row: dict[str, Any]) -> TokenizedExample:
    return TokenizedExample(
        input_ids=torch.as_tensor(row["input_ids"], dtype=torch.long).cuda(),
        attention_mask=torch.as_tensor(row["attention_mask"], dtype=torch.long).cuda(),
        label_id=int(row["label_id"]),
        example_id=int(row["example_id"]),
    )


def _render_token_spans(
    ex: dict[str, Any],
    values: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    hover_label: str,
) -> str:
    scale = max(float(np.abs(values).max()), 1e-12)
    spans = []
    for i, tid in enumerate(ex["input_ids"]):
        tok = html.escape(_visualize_invisibles(_decode_token(tokenizer, int(tid))))
        v = float(values[i])
        alpha = min(abs(v) / scale, 1.0)
        rgb = "255,80,80" if v > 0 else "80,80,255"
        spans.append(
            f"<span class='tok' style='background:rgba({rgb},{alpha:.2f})' "
            f"title='{hover_label}={v:+.4f}'>{tok}</span>"
        )
    return "".join(spans)


def _make_example_selector_widget(
    train_list: list[dict[str, Any]],
    probe_list: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    basis_list: list[BasisRow] | None,
    compute: Callable[[str, int, BasisRow | None], np.ndarray],
    hover_label: str,
) -> widgets.Widget:
    split_dd = widgets.Dropdown(options=["probe", "train"], value="probe", description="Split:")

    def current_list() -> list[dict[str, Any]]:
        return probe_list if split_dd.value == "probe" else train_list

    def example_options() -> list[tuple[str, int]]:
        return [(f"{i}: example {ex['example_id']}", i) for i, ex in enumerate(current_list())]

    example_dd: widgets.Dropdown = widgets.Dropdown(
        description="Example:", options=example_options(), value=0,
    )
    selectors: list[widgets.Widget] = [split_dd, example_dd]
    basis_dd: widgets.Dropdown | None = None
    if basis_list is not None:
        basis_dd = widgets.Dropdown(
            options=[(f"cluster {b['cluster_id']} (ex {b['example_id']})", i) for i, b in enumerate(basis_list)],
            value=0,
            description="Basis:",
        )
        selectors.append(basis_dd)
    output = widgets.VBox([])

    def update(*_: Any) -> None:
        if example_dd.value is None:
            return
        if basis_dd is not None and basis_dd.value is None:
            return
        ex_idx = int(example_dd.value)  # type: ignore[arg-type]
        ex = current_list()[ex_idx]
        basis = (
            basis_list[int(basis_dd.value)]  # type: ignore[arg-type]
            if basis_list is not None and basis_dd is not None
            else None
        )
        values = compute(str(split_dd.value), ex_idx, basis)
        answer = html.escape(_decode_token(tokenizer, int(ex["label_id"])).strip())
        output.children = (widgets.HTML(
            f"{_TOKEN_STYLE}"
            f"<p>example {ex['example_id']} &middot; correct: <b>{answer}</b></p>"
            f"<pre style='white-space:pre-wrap;font-family:monospace'>"
            f"{_render_token_spans(ex, values, tokenizer, hover_label)}</pre>"
        ),)

    def on_split_change(_: Any) -> None:
        example_dd.unobserve(update, names="value")
        example_dd.options = example_options()
        example_dd.value = 0
        example_dd.observe(update, names="value")
        update()

    split_dd.observe(on_split_change, names="value")
    example_dd.observe(update, names="value")
    if basis_dd is not None:
        basis_dd.observe(update, names="value")
    update()

    return widgets.VBox([widgets.HBox(selectors), output])


def example_fingerprint_widget(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
    alpha: float = ALPHA_DEFAULT,
) -> widgets.Widget:
    basis_list = _load_basis(cache_path)
    train_list = list(_load(cache_path / "tok_train"))
    probe_list = list(_load(cache_path / "tok_probe"))
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    probe_bounds = np.cumsum([0, *(len(ex["input_ids"]) for ex in probe_list)])
    baseline_fn = make_baseline_fn(model)

    def compute(split: str, ex_idx: int, basis: BasisRow | None) -> np.ndarray:
        assert basis is not None
        if split == "probe":
            fp = np.asarray(basis["loss_fingerprint"], dtype=np.float32)
            return fp[probe_bounds[ex_idx] : probe_bounds[ex_idx + 1]]
        tokenized = _to_tokenized(train_list[ex_idx])
        baseline = baseline_fn(tokenized)
        v = torch.as_tensor(basis["vector"], dtype=torch.float32).cuda()
        loss_delta, _ = compute_delta(
            model, tokenized, baseline, layer, v, float(basis["std_dev"]), alpha
        )
        return loss_delta.detach().float().cpu().numpy()

    return _make_example_selector_widget(train_list, probe_list, tokenizer, basis_list, compute, "Δloss")


def example_cosine_widget(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
) -> widgets.Widget:
    basis_list = _load_basis(cache_path)
    train_list = list(_load(cache_path / "tok_train"))
    probe_list = list(_load(cache_path / "tok_probe"))
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    def compute(split: str, ex_idx: int, basis: BasisRow | None) -> np.ndarray:
        assert basis is not None
        tokenized = _to_tokenized((probe_list if split == "probe" else train_list)[ex_idx])
        neg_grad = compute_neg_gradients(model, tokenized, layer)
        v = torch.as_tensor(basis["vector"], dtype=neg_grad.dtype, device=neg_grad.device)
        cos = F.cosine_similarity(neg_grad, v.unsqueeze(0), dim=-1)
        return cos.detach().float().cpu().numpy()

    return _make_example_selector_widget(train_list, probe_list, tokenizer, basis_list, compute, "cos")


def example_gradient_norm_widget(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
) -> widgets.Widget:
    train_list = list(_load(cache_path / "tok_train"))
    probe_list = list(_load(cache_path / "tok_probe"))
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    def compute(split: str, ex_idx: int, _basis: BasisRow | None) -> np.ndarray:
        tokenized = _to_tokenized((probe_list if split == "probe" else train_list)[ex_idx])
        neg_grad = compute_neg_gradients(model, tokenized, layer)
        return neg_grad.norm(dim=-1).detach().float().cpu().numpy()

    return _make_example_selector_widget(train_list, probe_list, tokenizer, None, compute, "‖∇‖")
