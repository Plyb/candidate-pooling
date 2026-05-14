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
from datasets import Dataset, load_from_disk
from IPython.display import display
from transformers import PreTrainedTokenizerBase

from byutils import load_tokenizer

from candidate_pooling.lib.typed_dataset import TypedDataset
from candidate_pooling.pipeline import MODEL_ID


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
            tok = html.escape(_decode_token(tokenizer, int(tid)))
            style = "background:#ffeb3b;font-weight:bold" if i == token_pos else ""
            spans.append(f"<span style='{style}'>{tok}</span>")
        answer = html.escape(_decode_token(tokenizer, int(ex["label_id"])))
        return widgets.HTML(
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
                tok = html.escape(_decode_token(tokenizer, int(tid)))
                d = float(deltas[i])
                alpha = min(abs(d) / scale, 1.0)
                rgb = "255,80,80" if d > 0 else "80,80,255"
                spans.append(
                    f"<span style='background:rgba({rgb},{alpha:.2f})' "
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
        return widgets.HTML("<hr>".join(parts))

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
