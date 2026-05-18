import html
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from datasets import Dataset, load_from_disk
from IPython.display import display
from nnsight import LanguageModel
from transformers import PreTrainedTokenizerBase
from jaxtyping import Float

from byutils import load_tokenizer

from candidate_pooling.fingerprint import (
    ALPHA_DEFAULT,
    _final_token_loss_entropy,
    compute_jacobians,
    make_baseline_fn,
)
from candidate_pooling.lib.typed_dataset import TypedDataset
from candidate_pooling.mining import LAYER, compute_neg_gradients
from candidate_pooling.pipeline import MODEL_ID
from candidate_pooling.types import BasisDirection, Candidate, TokenizedExample, to_transformer_input



def _load(path: Path) -> TypedDataset[dict[str, Any]]:
    return TypedDataset[dict[str, Any]](cast(Dataset, load_from_disk(str(path))))

def _load_basis(cache_path: Path) -> list[BasisDirection]:
    return list(_load(cache_path / "out")) # type: ignore[return-type]

def _load_tok_train(cache_path: Path) -> TypedDataset[TokenizedExample]:
    return cast(TypedDataset[TokenizedExample], _load(cache_path / "tok_train"))

def _load_tok_probe(cache_path: Path) -> TypedDataset[TokenizedExample]:
    return cast(TypedDataset[TokenizedExample], _load(cache_path / "tok_probe"))

def _load_mined(cache_path: Path) -> TypedDataset[Candidate]:
    return cast(TypedDataset[Candidate], _load(cache_path / "mined"))


def _basis_dropdown_with_container(
    basis_dir_list: list[BasisDirection],
    render: Callable[[BasisDirection], widgets.Widget],
) -> widgets.Widget:
    options = [(f"Direction {b['cluster_id']} (ex {b['example_id']})", i) for i, b in enumerate(basis_dir_list)]
    dropdown = widgets.Dropdown(options=options, description="Basis:")
    container = widgets.VBox([])

    def show(idx: int) -> None:
        container.children = (render(basis_dir_list[idx]),)

    dropdown.observe(lambda change: show(int(change["new"])), names="value")
    show(int(dropdown.value))  # type: ignore[arg-type]
    return widgets.VBox([dropdown, container])


def _decode_token(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    decoded = tokenizer.decode([token_id])
    return decoded if isinstance(decoded, str) else "".join(decoded)


def _visualize_invisibles(token: str) -> str:
    return token.replace("\n", "↵\n").replace("\t", "→\t").replace("\r", "␍")


_TOKEN_STYLE = "<style>.tok:hover { outline: 1px solid #000; }</style>"

def _tok_str(tokenizer: PreTrainedTokenizerBase, tok_id: int) -> str: 
    return html.escape(_visualize_invisibles(_decode_token(tokenizer, tok_id)))


def source_example_widget(cache_path: Path) -> widgets.Widget:
    basis_list = _load_basis(cache_path)
    train_by_id = {row["example_id"]: row for row in _load_tok_train(cache_path)}
    mined_by_id: dict[int, list[Candidate]] = {}
    for cand in _load_mined(cache_path):
        mined_by_id.setdefault(cand["example_id"], []).append(cand)
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    def find_token_pos(basis: BasisDirection) -> int:
        target = np.asarray(basis["vector"], dtype=np.float32)
        for cand in mined_by_id[basis["example_id"]]:
            if np.allclose(np.asarray(cand["vector"], dtype=np.float32), target, atol=1e-6):
                return int(cand["token_pos"])
        raise ValueError(f"no mined candidate matched basis for example {basis['example_id']}")

    def render(basis: BasisDirection) -> widgets.Widget:
        ex = train_by_id[basis["example_id"]]
        token_pos = find_token_pos(basis)
        spans = []
        for i, tok_id in enumerate(ex["input_ids"]):
            tok = _tok_str(tokenizer, int(tok_id))
            style = "background:#ffeb3b;font-weight:bold" if i == token_pos else ""
            spans.append(f"<span class='tok' style='{style}'>{tok}</span>")
        answer = _tok_str(tokenizer, int(ex["label_id"]))
        return widgets.HTML(
            f"{_TOKEN_STYLE}"
            f"<pre style='white-space:pre-wrap;font-family:monospace'>{''.join(spans)}</pre>"
            f"<p><b>Correct answer:</b> {answer}</p>"
        )

    return _basis_dropdown_with_container(basis_list, render)


def top_probe_examples_widget(cache_path: Path, k: int = 10) -> widgets.Widget:
    basis_list = _load_basis(cache_path)
    probe_list = list(_load_tok_probe(cache_path))
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    lengths = [len(ex["input_ids"]) for ex in probe_list]
    bounds = np.cumsum([0, *lengths])

    def split_by_example(fingerprint: Float[Tensor, "total_probe_tokens"]) -> list[Float[np.ndarray, "seq"]]:
        arr = np.asarray(fingerprint, dtype=np.float32)
        return [arr[bounds[i] : bounds[i + 1]] for i in range(len(probe_list))]

    def render(basis: BasisDirection) -> widgets.Widget:
        loss_deltas_per_example = split_by_example(basis["loss_fingerprint"])
        min_delta = np.array([float(p.min()) for p in loss_deltas_per_example])
        bottom_idx = np.argsort(min_delta)[:k]
        parts: list[str] = []
        for ranking, idx in enumerate(bottom_idx):
            ex = probe_list[int(idx)]
            deltas = loss_deltas_per_example[int(idx)]
            scale = max(float(np.abs(deltas).max()), 1e-12)
            spans = []
            for i, tok_id in enumerate(ex["input_ids"]):
                tok = _tok_str(tokenizer, int(tok_id))
                delta = float(deltas[i])

                transparency = min(abs(delta) / scale, 1.0)
                rgb = "255,80,80" if delta > 0 else "80,80,255"

                spans.append(
                    f"<span class='tok' style='background:rgba({rgb},{transparency:.2f})' "
                    f"title='Δloss={delta:+.4f}'>{tok}</span>"
                )
            answer = _tok_str(tokenizer, int(ex["label_id"]))
            parts.append(
                f"<div style='margin-bottom:0.5em'>"
                f"<b>#{ranking + 1}</b> example {ex['example_id']} &middot; "
                f"min Δloss={min_delta[idx]:+.4f} &middot; correct: <b>{answer}</b>"
                f"<pre style='white-space:pre-wrap;font-family:monospace;margin:0.25em 0'>"
                f"{''.join(spans)}</pre></div>"
            )
        return widgets.HTML(_TOKEN_STYLE + "<hr/>".join(parts))

    return _basis_dropdown_with_container(basis_list, render)


def fingerprint_histograms_widget(cache_path: Path) -> widgets.Widget:
    basis_list = _load_basis(cache_path)
    probe_list = _load_tok_probe(cache_path)
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    per_token_answer: list[str] = []
    for ex in probe_list:
        letter = _decode_token(tokenizer, int(ex["label_id"]))
        per_token_answer.extend([letter] * len(ex["input_ids"]))
    answer_arr = np.asarray(per_token_answer)
    answer_labels = sorted(set(per_token_answer))

    def render(basis: BasisDirection) -> widgets.Widget:
        out = widgets.Output()
        with out:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            for ax, key, label in [
                (axs[0], "loss_fingerprint", "Δloss"),
                (axs[1], "entropy_fingerprint", "Δentropy"),
            ]:
                arr = np.asarray(basis[key], dtype=np.float32)
                edges = np.histogram_bin_edges(arr, bins=50)
                zero_bin = int(np.clip(np.searchsorted(edges, 0.0, side="right") - 1, 0, len(edges) - 2))
                low, high = float(edges[zero_bin]), float(edges[zero_bin + 1])
                mask = (arr < low) | (arr >= high)
                kept = arr[mask]
                kept_density = len(kept) / max(len(arr), 1)
                groups = [kept[answer_arr[mask] == a] for a in answer_labels]
                ax.hist(groups, bins=50, stacked=True, label=answer_labels)
                ax.set_title(f"{label} (kept: {kept_density:.2%}, excluded [{low:+.3g}, {high:+.3g}))")
                ax.set_xlabel(label)
                ax.legend(title="correct", fontsize=8)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
        return out

    return _basis_dropdown_with_container(basis_list, render)


def _render_token_spans(
    ex: TokenizedExample,
    values: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    hover_label: str,
) -> str:
    scale = max(float(np.abs(values).max()), 1e-12)
    spans = []
    for i, tok_id in enumerate(ex["input_ids"]):
        tok = _tok_str(tokenizer, int(tok_id))
        v = float(values[i])
        alpha = min(abs(v) / scale, 1.0)
        rgb = "255,80,80" if v > 0 else "80,80,255"
        spans.append(
            f"<span class='tok' style='background:rgba({rgb},{alpha:.2f})' "
            f"title='{hover_label}={v:+.4f}'>{tok}</span>"
        )
    return "".join(spans)

# TODO: consider Reacton

# TODO: unify this with _basis_dropdown_with_container
def _make_example_selector_widget( # TODO: is there a way to factor out the list-of-dropdowns aspect of this? might need to do some overloads to get the different arities
    train_list: list[TokenizedExample],
    probe_list: list[TokenizedExample],
    tokenizer: PreTrainedTokenizerBase,
    basis_list: list[BasisDirection] | None,
    compute: Callable[[str, int, BasisDirection | None], np.ndarray],
    hover_label: str,
) -> widgets.Widget:
    split_dd = widgets.Dropdown(options=["probe", "train"], value="probe", description="Split:")

    def current_list() -> list[TokenizedExample]:
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
        answer = _tok_str(tokenizer, int(ex["label_id"]))
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
) -> widgets.Widget:
    basis_list = _load_basis(cache_path)
    train_list = list(_load_tok_train(cache_path))
    probe_list = list(_load_tok_probe(cache_path))
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    def compute(split: str, ex_idx: int, basis: BasisDirection | None) -> np.ndarray:
        assert basis is not None
        tokenized = (probe_list if split == "probe" else train_list)[ex_idx]
        jac_loss, _ = compute_jacobians(model, tokenized, layer)
        v = torch.as_tensor(basis["vector"], dtype=jac_loss.dtype, device=jac_loss.device)
        return (jac_loss @ v).detach().float().cpu().numpy()

    return _make_example_selector_widget(train_list, probe_list, tokenizer, basis_list, compute, "J·v")


def example_fingerprint_widget_steered(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
    alpha: float = ALPHA_DEFAULT,
) -> widgets.Widget:
    """For each token t: steer the layer-`layer` activation by `alpha * v`
    and render the resulting Δloss on the final token. Tests whether the per-token Jacobian
    score is self-consistent under actual steering at that magnitude."""
    basis_list = _load_basis(cache_path)
    train_list = list(_load_tok_train(cache_path))
    probe_list = list(_load_tok_probe(cache_path))
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]
    baseline_fn = make_baseline_fn(model)

    def compute(split: str, ex_idx: int, basis: BasisDirection | None) -> np.ndarray:
        assert basis is not None
        tokenized = (probe_list if split == "probe" else train_list)[ex_idx]
        v = torch.as_tensor(basis["vector"], device="cuda")
        baseline_loss = baseline_fn(tokenized)["loss"].cuda()
        seq = int(tokenized["input_ids"].shape[0])
        deltas = torch.empty(seq, device="cuda")
        for t in range(seq):
            perturbation = alpha * v
            with torch.no_grad(), model.trace(to_transformer_input(tokenized)):
                model.model.layers[layer].output[0, t] += perturbation  # type: ignore[attr-defined]
                logits = model.output.logits.save()  # type: ignore[attr-defined]
            loss_t, _ = _final_token_loss_entropy(logits[0], tokenized["label_id"])
            deltas[t] = loss_t - baseline_loss
        return deltas.detach().float().cpu().numpy()

    return _make_example_selector_widget(train_list, probe_list, tokenizer, basis_list, compute, "Δloss(steered)")


def example_steering_curve_widget(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
    alpha_max: float = 1.0,
    n_alphas: int = 21,
) -> widgets.Widget: # TODO: this could use a similar dropdown widget, but further abstracted so that its content can be anything
    """Pick a (split, example, basis, token) and sweep α: plot final-token loss vs α
    when the layer-`layer` activation at the chosen token is perturbed by α·v. Overlays
    the linear prediction baseline + α·(J_loss[t]·v) for direct comparison."""
    basis_list = _load_basis(cache_path)
    train_list = list(_load_tok_train(cache_path))
    probe_list = list(_load_tok_probe(cache_path))
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]
    baseline_fn = make_baseline_fn(model)

    split_dd = widgets.Dropdown(options=["probe", "train"], value="probe", description="Split:")
    basis_dd = widgets.Dropdown(
        description="Basis:",
        options=[(f"cluster {b['cluster_id']} (ex {b['example_id']})", i) for i, b in enumerate(basis_list)],
        value=0,
    )
    example_dd: widgets.Dropdown = widgets.Dropdown(description="Example:")
    token_dd: widgets.Dropdown = widgets.Dropdown(description="Token:")
    alpha_max_box = widgets.FloatText(value=alpha_max, description="α max:", layout=widgets.Layout(width="180px"))
    n_alphas_box = widgets.IntText(value=n_alphas, description="n α:", layout=widgets.Layout(width="160px"))
    log_toggle = widgets.Checkbox(value=False, description="log α", indent=False)
    output = widgets.Output()

    def current_list() -> list[TokenizedExample]:
        return probe_list if split_dd.value == "probe" else train_list

    def example_options() -> list[tuple[str, int]]:
        return [(f"{i}: example {ex['example_id']}", i) for i, ex in enumerate(current_list())]

    def token_options(ex: TokenizedExample) -> list[tuple[str, int]]:
        return [
            (f"{i}: {_visualize_invisibles(_decode_token(tokenizer, int(tok_id)))!r}", i)
            for i, tok_id in enumerate(ex["input_ids"])
        ]

    jac_cache: dict[
        tuple[str, int],
        tuple[Float[Tensor, "seq d_model"], Float[Tensor, "seq d_model"], TokenizedExample]
    ] = {}

    def get_cached(split: str, ex_idx: int) -> tuple[Float[Tensor, "seq d_model"], Float[Tensor, "seq d_model"], TokenizedExample]:
        key = (split, ex_idx)
        if key not in jac_cache:
            tokenized = current_list()[ex_idx]
            jac_loss, _ = compute_jacobians(model, tokenized, layer)
            baseline_loss = baseline_fn(tokenized)["loss"].cuda().detach()
            jac_cache[key] = (jac_loss.detach(), baseline_loss, tokenized)
        return jac_cache[key]

    def alpha_array() -> np.ndarray:
        amax = float(alpha_max_box.value)
        n = max(int(n_alphas_box.value), 3)
        if log_toggle.value:
            n_side = max((n - 1) // 2, 1)
            small = max(amax * 1e-4, 1e-8)
            side = np.logspace(np.log10(small), np.log10(max(amax, small * 10)), n_side)
            return np.sort(np.concatenate([-side, [0.0], side]))
        return np.linspace(-amax, amax, n)

    def update(*_: Any) -> None:
        if example_dd.value is None or token_dd.value is None or basis_dd.value is None:
            return
        ex_idx = int(example_dd.value)  # type: ignore[arg-type]
        t = int(token_dd.value)  # type: ignore[arg-type]
        basis = basis_list[int(basis_dd.value)]  # type: ignore[arg-type]

        jac_loss, baseline_loss, tokenized = get_cached(str(split_dd.value), ex_idx)
        v = torch.as_tensor(basis["vector"], dtype=jac_loss.dtype, device=jac_loss.device)
        score = float(jac_loss[t] @ v)
        baseline_scalar = float(baseline_loss)

        alphas = alpha_array()
        losses = np.empty_like(alphas)
        for i, a in enumerate(alphas):
            if a == 0.0:
                losses[i] = baseline_scalar
                continue
            perturbation = float(a) * v
            with torch.no_grad(), model.trace(to_transformer_input(tokenized)):
                model.model.layers[layer].output[0, t] += perturbation  # type: ignore[attr-defined]
                logits = model.output.logits.save()  # type: ignore[attr-defined] #TODO factor out perturbed logit computation
            loss_t, _entropy = _final_token_loss_entropy(logits[0], tokenized["label_id"])
            losses[i] = float(loss_t)

        output.clear_output(wait=True)
        with output:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.axhline(baseline_scalar, color="gray", linestyle=":", linewidth=0.8, label="baseline")
            ax.plot(alphas, baseline_scalar + alphas * score, "--", color="C1", label=f"linear (J·v={score:+.3g})")
            ax.plot(alphas, losses, "o-", color="C0", markersize=4, label="actual")
            tok = _visualize_invisibles(_decode_token(tokenizer, int(tokenized["input_ids"][t].item())))
            ax.set_xlabel("α  (perturbation = α · v)")
            ax.set_ylabel("final-token loss")
            ax.set_title(f"example {tokenized['example_id']} · token {t}: {tok!r}")
            if log_toggle.value:
                linthresh = max(float(alpha_max_box.value) * 1e-4, 1e-8)
                ax.set_xscale("symlog", linthresh=linthresh)
            ax.legend(fontsize=8)
            fig.tight_layout()
            display(fig)
            plt.close(fig)

    def repopulate_tokens(*_: Any) -> None:
        if example_dd.value is None:
            return
        ex = current_list()[int(example_dd.value)]  # type: ignore[arg-type]
        token_dd.unobserve(update, names="value")
        token_dd.options = token_options(ex)
        token_dd.value = 0
        token_dd.observe(update, names="value")
        update()

    def on_split_change(*_: Any) -> None:
        example_dd.unobserve(repopulate_tokens, names="value")
        example_dd.options = example_options()
        example_dd.value = 0
        example_dd.observe(repopulate_tokens, names="value")
        repopulate_tokens()

    example_dd.options = example_options()
    example_dd.value = 0
    token_dd.options = token_options(current_list()[0])
    token_dd.value = 0

    split_dd.observe(on_split_change, names="value")
    example_dd.observe(repopulate_tokens, names="value")
    token_dd.observe(update, names="value")
    basis_dd.observe(update, names="value")
    alpha_max_box.observe(update, names="value")
    n_alphas_box.observe(update, names="value")
    log_toggle.observe(update, names="value")

    update()

    return widgets.VBox([
        widgets.HBox([split_dd, basis_dd, example_dd, token_dd]),
        widgets.HBox([alpha_max_box, n_alphas_box, log_toggle]),
        output,
    ])


def logit_lens_widget(
    cache_path: Path,
    model: LanguageModel,
    top_k: int = 20,
) -> widgets.Widget:
    basis_list = _load_basis(cache_path)
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]
    W_U: Float[Tensor, "vocab d_model"] = model.lm_head.weight.detach()  # type: ignore[attr-defined]

    def render(basis: BasisDirection) -> widgets.Widget:
        v = torch.as_tensor(basis["vector"], dtype=W_U.dtype, device=W_U.device)
        direct_effects = (W_U @ v).float().cpu().numpy()

        order = np.argsort(direct_effects)
        bot_idx = order[:top_k]
        top_idx = order[-top_k:][::-1]

        def fmt_table(indices: np.ndarray, title: str) -> str:
            rows = "".join(
                f"<tr><td style='padding:0 0.5em'>{rank + 1}</td>"
                f"<td style='font-family:monospace;padding:0 0.5em'>{_tok_str(tokenizer, int(idx))}</td>"
                f"<td style='text-align:right;padding:0 0.5em'>{float(direct_effects[idx]):+.4f}</td></tr>"
                for rank, idx in enumerate(indices)
            )
            return (
                f"<div><h4 style='margin:0 0 0.25em 0'>{title}</h4>"
                f"<table style='border-collapse:collapse;font-size:0.9em'>"
                f"<thead><tr><th>#</th><th style='text-align:left'>token</th>"
                f"<th style='text-align:right'>W_U·v</th></tr></thead>"
                f"<tbody>{rows}</tbody></table></div>"
            )

        tables = widgets.HTML(
            f"<div style='display:flex;gap:2em'>"
            f"{fmt_table(top_idx, 'Top upweighted')}"
            f"{fmt_table(bot_idx, 'Top downweighted')}"
            f"</div>"
        )

        hist = widgets.Output()
        with hist:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.hist(direct_effects, bins=80)
            ax.axvline(0.0, color="gray", linewidth=0.6)
            ax.set_xlabel("direct logit effect  (W_U · v)")
            ax.set_ylabel("token count")
            ax.set_title(f"distribution over vocab (|V|={len(direct_effects)})")
            fig.tight_layout()
            display(fig)
            plt.close(fig)

        return widgets.VBox([tables, hist])

    return _basis_dropdown_with_container(basis_list, render)


def example_cosine_widget(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
) -> widgets.Widget:
    basis_list = _load_basis(cache_path)
    train_list = list(_load_tok_train(cache_path))
    probe_list = list(_load_tok_probe(cache_path))
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    def compute(split: str, ex_idx: int, basis: BasisDirection | None) -> np.ndarray:
        assert basis is not None
        tokenized = (probe_list if split == "probe" else train_list)[ex_idx]
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
    train_list = list(_load_tok_train(cache_path))
    probe_list = list(_load_tok_probe(cache_path))
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    def compute(split: str, ex_idx: int, _basis: BasisDirection | None) -> np.ndarray:
        tokenized = (probe_list if split == "probe" else train_list)[ex_idx]
        neg_grad = compute_neg_gradients(model, tokenized, layer)
        return neg_grad.norm(dim=-1).detach().float().cpu().numpy()

    return _make_example_selector_widget(train_list, probe_list, tokenizer, None, compute, "‖∇‖")
