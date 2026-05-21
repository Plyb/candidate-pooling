import base64
import html
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any, Hashable, Mapping, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import reacton
import reacton.ipywidgets as rw
import torch
import torch.nn.functional as F
from torch import Tensor
from datasets import Dataset, load_from_disk
from IPython.display import display
from nnsight import LanguageModel
from reacton.core import Element
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


@reacton.component
def _MatplotlibView(draw: Callable[[], None], redraw_key: Hashable) -> Element:
    out_el = rw.Output()

    def effect() -> None:
        out_widget = cast(widgets.Output, reacton.get_widget(out_el))
        out_widget.clear_output(wait=True)
        with out_widget:
            draw()

    reacton.use_effect(effect, [redraw_key])
    return out_el

@reacton.component
def _BasisDropdown(
    basis_dir_list: list[BasisDirection],
    render: Callable[[BasisDirection], Element],
) -> Element:
    idx, set_idx = reacton.use_state(0)
    options = [(f"Direction {b['cluster_id']} (ex {b['example_id']})", i) for i, b in enumerate(basis_dir_list)]
    with rw.VBox() as main: # TODO: not sure I love this `with` syntax. Can we do this more functionally?
        rw.Dropdown(options=options, value=idx, on_value=set_idx, description="Basis:")
        render(basis_dir_list[idx])
    return main


def _decode_token(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    decoded = tokenizer.decode([token_id])
    return decoded if isinstance(decoded, str) else "".join(decoded)


def _visualize_invisibles(token: str) -> str:
    return token.replace("\n", "↵\n").replace("\t", "→\t").replace("\r", "␍")


_TOKEN_STYLE = "<style>.tok:hover { outline: 1px solid #000; }</style>"

def _tok_str(tokenizer: PreTrainedTokenizerBase, tok_id: int) -> str:
    return html.escape(_visualize_invisibles(_decode_token(tokenizer, tok_id)))


def source_example_widget(cache_path: Path) -> Element:
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

    def render(basis: BasisDirection) -> Element:
        ex = train_by_id[basis["example_id"]]
        token_pos = find_token_pos(basis)
        spans = []
        for i, tok_id in enumerate(ex["input_ids"]):
            tok = _tok_str(tokenizer, int(tok_id))
            style = "background:#ffeb3b;font-weight:bold" if i == token_pos else ""
            spans.append(f"<span class='tok' style='{style}'>{tok}</span>")
        answer = _tok_str(tokenizer, int(ex["label_id"]))
        return rw.HTML(value=(
            f"{_TOKEN_STYLE}"
            f"<pre style='white-space:pre-wrap;font-family:monospace'>{''.join(spans)}</pre>"
            f"<p><b>Correct answer:</b> {answer}</p>"
        ))

    return _BasisDropdown(basis_list, render)


def top_probe_examples_widget(cache_path: Path, k: int = 10) -> Element:
    basis_list = _load_basis(cache_path)
    probe_list = list(_load_tok_probe(cache_path))
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    lengths = [len(ex["input_ids"]) for ex in probe_list]
    bounds = np.cumsum([0, *lengths])

    def split_by_example(fingerprint: Float[Tensor, "total_probe_tokens"]) -> list[Float[np.ndarray, "seq"]]:
        arr = np.asarray(fingerprint, dtype=np.float32)
        return [arr[bounds[i] : bounds[i + 1]] for i in range(len(probe_list))]

    def render(basis: BasisDirection) -> Element:
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
        return rw.HTML(value=_TOKEN_STYLE + "<hr/>".join(parts))

    return _BasisDropdown(basis_list, render)


def fingerprint_histograms_widget(cache_path: Path) -> Element:
    basis_list = _load_basis(cache_path)
    probe_list = _load_tok_probe(cache_path)
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    per_token_answer: list[str] = []
    for ex in probe_list:
        letter = _decode_token(tokenizer, int(ex["label_id"]))
        per_token_answer.extend([letter] * len(ex["input_ids"]))
    answer_arr = np.asarray(per_token_answer)
    answer_labels = sorted(set(per_token_answer))

    def render(basis: BasisDirection) -> Element:
        def draw() -> None:
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

        return _MatplotlibView(draw=draw, redraw_key=basis["cluster_id"])

    return _BasisDropdown(basis_list, render)


@reacton.component
def _TokenSelector(
    example: TokenizedExample,
    tokenizer: PreTrainedTokenizerBase,
    selected_idx: int,
    on_select: Callable[[int], None],
    values: np.ndarray | None = None,
    hover_label: str = "",
) -> Element:
    scale = max(float(np.abs(values).max()), 1e-12) if values is not None else 1.0
    with rw.HBox(layout={"flex_flow": "row wrap", "align_items": "center"}) as main:
        for i, tok_id in enumerate(example["input_ids"]):
            raw = _decode_token(tokenizer, int(tok_id))
            tok_display = _visualize_invisibles(raw)
            if values is not None:
                v = float(values[i])
                alpha = min(abs(v) / scale, 1.0)
                base = "255,80,80" if v > 0 else "80,80,255"
                bg = f"rgba({base},{alpha:.2f})"
                tooltip = f"{i}: {tok_display!r} · {hover_label}={v:+.4f}" if hover_label else f"{i}: {tok_display!r} · {v:+.4f}"
            else:
                bg = "white"
                tooltip = f"{i}: {tok_display!r}"
            border = "2px solid black" if i == selected_idx else "1px solid #ddd"
            rw.Button(
                description=tok_display.replace(" ", " "),
                on_click=lambda i=i: on_select(i),
                layout={
                    "width": "auto",
                    "min_width": "0",
                    "height": "auto",
                    "padding": "0 4px",
                    "margin": "1px",
                    "border": border,
                },
                style={"button_color": bg, "font_family": "monospace", "font_size": "0.9em"},
                tooltip=tooltip,
            )
            if "\n" in raw:
                rw.Box(layout={"flex": "0 0 100%", "height": "0px"})
    return main


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

# TODO unify with _BasisDropdown
# TODO: is there a way to factor out the list-of-dropdowns aspect of this? might need to do some overloads to get the different arities
@reacton.component
def _ExampleSelector(
    example_splits: Mapping[str, list[TokenizedExample]],
    tokenizer: PreTrainedTokenizerBase,
    basis_list: list[BasisDirection] | None,
    compute: Callable[[TokenizedExample, BasisDirection | None], np.ndarray],
    hover_label: str,
) -> Element:
    split_keys = list(example_splits.keys())
    split, set_split = reacton.use_state(split_keys[0])
    example_idx, set_example_idx = reacton.use_state(0)
    basis_idx, set_basis_idx = reacton.use_state(0)

    current_list = example_splits[split]
    example_idx = min(example_idx, len(current_list) - 1)
    ex = current_list[example_idx]
    basis = basis_list[basis_idx] if basis_list is not None else None

    def on_split(new_split: str) -> None:
        set_split(new_split)
        set_example_idx(0)

    values = compute(ex, basis)
    answer = _tok_str(tokenizer, int(ex["label_id"]))
    content_html = (
        f"{_TOKEN_STYLE}"
        f"<p>example {ex['example_id']} &middot; correct: <b>{answer}</b></p>"
        f"<pre style='white-space:pre-wrap;font-family:monospace'>"
        f"{_render_token_spans(ex, values, tokenizer, hover_label)}</pre>"
    )

    with rw.VBox() as main:
        with rw.HBox():
            rw.Dropdown(options=split_keys, value=split, on_value=on_split, description="Split:")
            rw.Dropdown(
                options=[(f"{i}: example {e['example_id']}", i) for i, e in enumerate(current_list)],
                value=example_idx,
                on_value=set_example_idx,
                description="Example:",
            )
            if basis_list is not None:
                rw.Dropdown(
                    options=[(f"cluster {b['cluster_id']} (ex {b['example_id']})", i) for i, b in enumerate(basis_list)],
                    value=basis_idx,
                    on_value=set_basis_idx,
                    description="Basis:",
                )
        rw.HTML(value=content_html)
    return main


def example_fingerprint_widget(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
    example_splits: Mapping[str, list[TokenizedExample]] | None = None,
) -> Element:
    basis_list = _load_basis(cache_path)
    example_splits = example_splits or {
        "train": list(_load_tok_train(cache_path)),
        "probe": list(_load_tok_probe(cache_path))
    }
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    def compute(example: TokenizedExample, basis: BasisDirection | None) -> np.ndarray:
        assert basis is not None
        jac_loss, _ = compute_jacobians(model, example, layer)
        v = torch.as_tensor(basis["vector"], dtype=jac_loss.dtype, device=jac_loss.device)
        return (jac_loss @ v).detach().float().cpu().numpy()

    return _ExampleSelector(example_splits, tokenizer, basis_list, compute, "J·v")


def example_activation_dot_widget(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
    example_splits: Mapping[str, list[TokenizedExample]] | None = None,
    exclude_bos: bool = False,
) -> Element:
    basis_list = _load_basis(cache_path)
    example_splits = example_splits or {
        "train": list(_load_tok_train(cache_path)),
        "probe": list(_load_tok_probe(cache_path))
    }
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    def compute(example: TokenizedExample, basis: BasisDirection | None) -> np.ndarray:
        assert basis is not None
        with torch.no_grad(), model.trace(to_transformer_input(example)):
            hidden = model.model.layers[layer].output[0].save()  # type: ignore[attr-defined]
        h: Float[Tensor, "seq d_model"] = hidden
        if exclude_bos:
            h[0, :] = 0.0
        v = torch.as_tensor(basis["vector"], dtype=h.dtype, device=h.device)
        return (h @ v).detach().float().cpu().numpy()

    return _ExampleSelector(example_splits, tokenizer, basis_list, compute, "h·v")


def example_fingerprint_widget_steered(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
    alpha: float = ALPHA_DEFAULT,
    example_splits: Mapping[str, list[TokenizedExample]] | None = None,
) -> Element:
    """For each token t: steer the layer-`layer` activation by `alpha * v`
    and render the resulting Δloss on the final token. Tests whether the per-token Jacobian
    score is self-consistent under actual steering at that magnitude."""
    basis_list = _load_basis(cache_path)
    example_splits = example_splits or {
        "train": list(_load_tok_train(cache_path)),
        "probe": list(_load_tok_probe(cache_path))
    }
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]
    baseline_fn = make_baseline_fn(model)

    def compute(example: TokenizedExample, basis: BasisDirection | None) -> np.ndarray:
        assert basis is not None
        v = torch.as_tensor(basis["vector"], device="cuda")
        baseline_loss = baseline_fn(example)["loss"].cuda()
        seq = int(example["input_ids"].shape[0])
        deltas = torch.empty(seq, device="cuda")
        for t in range(seq):
            perturbation = alpha * v
            with torch.no_grad(), model.trace(to_transformer_input(example)):
                model.model.layers[layer].output[0, t] += perturbation  # type: ignore[attr-defined]
                logits = model.output.logits.save()  # type: ignore[attr-defined]
            loss_t, _ = _final_token_loss_entropy(logits[0], example["label_id"])
            deltas[t] = loss_t - baseline_loss
        return deltas.detach().float().cpu().numpy()

    return _ExampleSelector(example_splits, tokenizer, basis_list, compute, "Δloss(steered)")


_JacCacheEntry = tuple[Float[Tensor, "seq d_model"], Float[Tensor, "seq d_model"], TokenizedExample]


@reacton.component
def _SteeringCurve(
    basis_list: list[BasisDirection],
    splits: Mapping[str, list[TokenizedExample]],
    tokenizer: PreTrainedTokenizerBase,
    model: LanguageModel,
    layer: int,
    alpha_max_default: float,
    n_alphas_default: int,
) -> Element:
    split_keys = list(splits.keys())
    split, set_split = reacton.use_state(split_keys[0])
    basis_idx, set_basis_idx = reacton.use_state(0)
    example_idx, set_example_idx = reacton.use_state(0)
    token_idx, set_token_idx = reacton.use_state(0)
    alpha_max, set_alpha_max = reacton.use_state(alpha_max_default)
    n_alphas, set_n_alphas = reacton.use_state(n_alphas_default)
    log_alpha, set_log_alpha = reacton.use_state(False)

    jac_cache_ref = reacton.use_ref(cast(dict[tuple[str, int], _JacCacheEntry], {}))
    baseline_fn = reacton.use_memo(lambda: make_baseline_fn(model), [id(model)])

    current_list = splits[split]
    example_idx = min(example_idx, len(current_list) - 1)
    tokenized_preview = current_list[example_idx]
    token_idx = min(token_idx, int(tokenized_preview["input_ids"].shape[0]) - 1)

    def on_split(new_split: str) -> None:
        set_split(new_split)
        set_example_idx(0)
        set_token_idx(0)

    def on_example(new_idx: int) -> None:
        set_example_idx(int(new_idx))
        set_token_idx(0)

    key = (split, example_idx)
    cache = jac_cache_ref.current
    if key not in cache:
        tokenized = current_list[example_idx]
        jac_loss, _ = compute_jacobians(model, tokenized, layer)
        baseline_loss = baseline_fn(tokenized)["loss"].cuda().detach()
        cache[key] = (jac_loss.detach(), baseline_loss, tokenized)
    jac_loss, baseline_loss, tokenized = cache[key]

    basis = basis_list[basis_idx]
    v = torch.as_tensor(basis["vector"], dtype=jac_loss.dtype, device=jac_loss.device)
    jv_per_token = (jac_loss @ v).detach().float().cpu().numpy()
    score = float(jv_per_token[token_idx])
    baseline_scalar = float(baseline_loss)

    if log_alpha:
        n_side = max((n_alphas - 1) // 2, 1)
        small = max(alpha_max * 1e-4, 1e-8)
        side = np.logspace(np.log10(small), np.log10(max(alpha_max, small * 10)), n_side)
        alphas = np.sort(np.concatenate([-side, [0.0], side]))
    else:
        alphas = np.linspace(-alpha_max, alpha_max, max(n_alphas, 3))

    losses = np.empty_like(alphas)
    for i, a in enumerate(alphas):
        if a == 0.0:
            losses[i] = baseline_scalar
            continue
        perturbation = float(a) * v
        with torch.no_grad(), model.trace(to_transformer_input(tokenized)):
            model.model.layers[layer].output[0, token_idx] += perturbation  # type: ignore[attr-defined]
            logits = model.output.logits.save()  # type: ignore[attr-defined]
        loss_t, _entropy = _final_token_loss_entropy(logits[0], tokenized["label_id"])
        losses[i] = float(loss_t)

    tok_label = _visualize_invisibles(_decode_token(tokenizer, int(tokenized["input_ids"][token_idx].item())))

    def draw() -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(baseline_scalar, color="gray", linestyle=":", linewidth=0.8, label="baseline")
        ax.plot(alphas, baseline_scalar + alphas * score, "--", color="C1", label=f"linear (J·v={score:+.3g})")
        ax.plot(alphas, losses, "o-", color="C0", markersize=4, label="actual")
        ax.set_xlabel("α  (perturbation = α · v)")
        ax.set_ylabel("final-token loss")
        ax.set_title(f"example {tokenized['example_id']} · token {token_idx}: {tok_label!r}")
        if log_alpha:
            linthresh = max(alpha_max * 1e-4, 1e-8)
            ax.set_xscale("symlog", linthresh=linthresh)
        ax.legend(fontsize=8)
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    redraw_key = (split, basis_idx, example_idx, token_idx, alpha_max, n_alphas, log_alpha)

    with rw.VBox() as main:
        with rw.HBox():
            rw.Dropdown(options=split_keys, value=split, on_value=on_split, description="Split:")
            rw.Dropdown(
                options=[(f"cluster {b['cluster_id']} (ex {b['example_id']})", i) for i, b in enumerate(basis_list)],
                value=basis_idx, on_value=set_basis_idx, description="Basis:",
            )
            rw.Dropdown(
                options=[(f"{i}: example {e['example_id']}", i) for i, e in enumerate(current_list)],
                value=example_idx, on_value=on_example, description="Example:",
            )
        _TokenSelector(
            example=tokenized,
            tokenizer=tokenizer,
            selected_idx=token_idx,
            on_select=set_token_idx,
            values=jv_per_token,
            hover_label="J·v",
        )
        with rw.HBox():
            rw.FloatText(value=alpha_max, on_value=set_alpha_max, description="α max:",
                         layout={"width": "180px"})
            rw.IntText(value=n_alphas, on_value=set_n_alphas, description="n α:",
                       layout={"width": "160px"})
            rw.Checkbox(value=log_alpha, on_value=set_log_alpha, description="log α", indent=False)
        _MatplotlibView(draw=draw, redraw_key=redraw_key)
    return main


def example_steering_curve_widget(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
    alpha_max: float = 1.0,
    n_alphas: int = 21,
) -> Element:
    """Pick a (split, example, basis, token) and sweep α: plot final-token loss vs α
    when the layer-`layer` activation at the chosen token is perturbed by α·v. Overlays
    the linear prediction baseline + α·(J_loss[t]·v) for direct comparison."""
    basis_list = _load_basis(cache_path)
    splits: Mapping[str, list[TokenizedExample]] = {
        "probe": list(_load_tok_probe(cache_path)),
        "train": list(_load_tok_train(cache_path)),
    }
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    return _SteeringCurve(
        basis_list=basis_list,
        splits=splits,
        tokenizer=tokenizer,
        model=model,
        layer=layer,
        alpha_max_default=alpha_max,
        n_alphas_default=n_alphas,
    )


def _capture_downstream_attn(
    model: LanguageModel,
    example: TokenizedExample,
    steer_layer: int,
    steer_token_idx: int | None = None,
    perturbation: Float[Tensor, "d_model"] | None = None,
) -> Float[Tensor, "n_downstream n_heads seq seq"]:
    n_layers = len(model.model.layers)  # type: ignore[attr-defined]
    downstream = list(range(steer_layer + 1, n_layers))
    config = model.model.config  # type: ignore[attr-defined]
    prev_output_attentions = getattr(config, "output_attentions", False)
    config.output_attentions = True
    saved: list[Any] = []
    try:
        with torch.no_grad(), model.trace(to_transformer_input(example)):
            if perturbation is not None and steer_token_idx is not None:
                model.model.layers[steer_layer].output[0, steer_token_idx] += perturbation  # type: ignore[attr-defined]
            for L in downstream:
                saved.append(model.model.layers[L].self_attn.output.save())  # type: ignore[attr-defined]
    finally:
        config.output_attentions = prev_output_attentions
    stacked = []
    for L_idx, out in enumerate(saved):
        L = steer_layer + 1 + L_idx
        attn_weights = out[1] if isinstance(out, tuple) and len(out) >= 2 else None
        if not isinstance(attn_weights, torch.Tensor):
            raise RuntimeError(
                f"layer {L} returned no attention weights (got {type(out).__name__} → "
                f"{type(attn_weights).__name__}); model must be loaded with eager_attn=True"
            )
        stacked.append(attn_weights[0])  # drop batch dim → [n_heads, seq, seq]
    return torch.stack(stacked, dim=0)


def compute_attention_pattern_deltas(
    model: LanguageModel,
    example: TokenizedExample,
    steer_layer: int,
    token_idx: int,
    direction: Float[Tensor, "d_model"],
    alpha: float,
) -> Float[Tensor, "n_downstream n_heads seq seq"]:
    """Return steered − baseline attention patterns for every layer strictly after
    `steer_layer`. Requires the model loaded with attn_implementation="eager"."""
    baseline = _capture_downstream_attn(model, example, steer_layer)
    if alpha == 0.0:
        return torch.zeros_like(baseline)
    perturbation = (alpha * direction).to(baseline.dtype).to(baseline.device)
    steered = _capture_downstream_attn(model, example, steer_layer, token_idx, perturbation)
    return steered - baseline


def _heatmap_png(arr: np.ndarray, vmax: float, size_px: int) -> bytes:
    fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


_THUMB_PX = 48


def _thumb_style_block(class_to_png: dict[str, bytes]) -> str:
    rules = []
    for cls, png in class_to_png.items():
        b64 = base64.b64encode(png).decode("ascii")
        rules.append(
            f"button.jupyter-button.{cls} {{"
            f" background-image: url('data:image/png;base64,{b64}') !important;"
            f" background-size: 100% 100% !important;"
            f" background-repeat: no-repeat !important;"
            f" background-color: transparent !important;"
            f" min-width: {_THUMB_PX + 4}px !important;"
            f" max-width: {_THUMB_PX + 4}px !important;"
            f" min-height: {_THUMB_PX + 4}px !important;"
            f" max-height: {_THUMB_PX + 4}px !important;"
            f" padding: 0 !important;"
            f" margin: 1px !important;"
            f" flex: 0 0 auto !important;"
            f" box-sizing: border-box !important;"
            f"}}"
        )
    return "<style>" + "\n".join(rules) + "</style>"


@reacton.component
def _ThumbnailButton(
    class_name: str,
    selected: bool,
    tooltip: str,
    on_click: Callable[[], None],
) -> Element:
    border = "2px solid #000" if selected else "1px solid #ccc"
    btn_el = rw.Button(
        description="",
        tooltip=tooltip,
        on_click=lambda *_args: on_click(),  # type: ignore[arg-type]
        layout={"border": border},
    )
    class_ref = reacton.use_ref(cast(str | None, None))

    def attach() -> None:
        btn = reacton.get_widget(btn_el)
        prev = class_ref.current
        if prev is not None and prev != class_name:
            remove_class = getattr(btn, "remove_class", None)
            if callable(remove_class):
                remove_class(prev)
        add_class = getattr(btn, "add_class", None)
        if callable(add_class):
            add_class(class_name)
        class_ref.current = class_name

    reacton.use_effect(attach, [class_name])
    return btn_el



@reacton.component
def _AttentionPatternDelta(
    basis_list: list[BasisDirection],
    splits: Mapping[str, list[TokenizedExample]],
    tokenizer: PreTrainedTokenizerBase,
    model: LanguageModel,
    layer: int,
    alpha_default: float,
) -> Element:
    split_keys = list(splits.keys())
    split, set_split = reacton.use_state(split_keys[0])
    basis_idx, set_basis_idx = reacton.use_state(0)
    example_idx, set_example_idx = reacton.use_state(0)
    token_idx, set_token_idx = reacton.use_state(0)
    alpha, set_alpha = reacton.use_state(alpha_default)
    selected_cell, set_selected_cell = reacton.use_state(cast(tuple, ("overall",)))

    jac_cache_ref = reacton.use_ref(cast(dict[tuple[str, int], tuple[Float[Tensor, "seq d_model"], TokenizedExample]], {}))
    delta_cache_ref = reacton.use_ref(cast(dict[tuple[str, int, int, int, float], Float[Tensor, "n_down n_heads seq seq"]], {}))

    current_list = splits[split]
    example_idx = min(example_idx, len(current_list) - 1)
    tokenized_preview = current_list[example_idx]
    token_idx = min(token_idx, int(tokenized_preview["input_ids"].shape[0]) - 1)

    def on_split(new_split: str) -> None:
        set_split(new_split)
        set_example_idx(0)
        set_token_idx(0)
        set_selected_cell(("overall",))

    def on_example(new_idx: int) -> None:
        set_example_idx(int(new_idx))
        set_token_idx(0)
        set_selected_cell(("overall",))

    jac_key = (split, example_idx)
    jac_cache = jac_cache_ref.current
    if jac_key not in jac_cache:
        tokenized = current_list[example_idx]
        jac_loss, _ = compute_jacobians(model, tokenized, layer)
        jac_cache[jac_key] = (jac_loss.detach(), tokenized)
    jac_loss, tokenized = jac_cache[jac_key]

    basis = basis_list[basis_idx]
    v = torch.as_tensor(basis["vector"], dtype=jac_loss.dtype, device=jac_loss.device)
    jv_per_token = (jac_loss @ v).detach().float().cpu().numpy()

    delta_key = (split, example_idx, basis_idx, token_idx, float(alpha))
    delta_cache = delta_cache_ref.current
    if delta_key not in delta_cache:
        delta_cache[delta_key] = compute_attention_pattern_deltas(
            model, tokenized, layer, token_idx, v, float(alpha)
        ).detach()
    delta = delta_cache[delta_key]  # [n_down, n_heads, seq, seq]

    delta_np = delta.float().cpu().numpy()
    n_down, n_heads, seq, _ = delta_np.shape
    vmax = float(np.abs(delta_np).max())
    if vmax == 0.0:
        vmax = 1e-12

    overall_avg = delta_np.mean(axis=(0, 1))  # [seq, seq]
    layer_avgs = delta_np.mean(axis=1)  # [n_down, seq, seq]

    class_to_png: dict[str, bytes] = {}
    delta_hash = abs(hash((delta_key, selected_cell, float(vmax)))) % (10 ** 8)

    def thumb_class(slot: str) -> str:
        cls = f"attn-thumb-{delta_hash}-{slot}"
        return cls

    overall_class = thumb_class("overall")
    class_to_png[overall_class] = _heatmap_png(overall_avg, vmax, _THUMB_PX)
    layer_classes: list[str] = []
    head_classes: list[list[str]] = []
    for L_idx in range(n_down):
        lc = thumb_class(f"L{L_idx}avg")
        class_to_png[lc] = _heatmap_png(layer_avgs[L_idx], vmax, _THUMB_PX)
        layer_classes.append(lc)
        row: list[str] = []
        for h in range(n_heads):
            hc = thumb_class(f"L{L_idx}H{h}")
            class_to_png[hc] = _heatmap_png(delta_np[L_idx, h], vmax, _THUMB_PX)
            row.append(hc)
        head_classes.append(row)
    style_block = _thumb_style_block(class_to_png)

    def cell_for(sel: tuple) -> tuple[np.ndarray, str]:
        if sel[0] == "overall":
            return overall_avg, "overall avg (all downstream heads)"
        if sel[0] == "layer":
            L = int(sel[1])
            return layer_avgs[L - (layer + 1)], f"L{L} avg (heads {n_heads})"
        L, h = int(sel[1]), int(sel[2])
        return delta_np[L - (layer + 1), h], f"L{L} H{h}"

    expanded_arr, expanded_title = cell_for(selected_cell)
    seq_ids = [int(t) for t in tokenized["input_ids"]]
    tick_labels = [_visualize_invisibles(_decode_token(tokenizer, t)) for t in seq_ids]

    def draw_expanded() -> None:
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(expanded_arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(seq))
        ax.set_yticks(range(seq))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7, family="monospace")
        ax.set_yticklabels(tick_labels, fontsize=7, family="monospace")
        ax.set_xticks(np.arange(-0.5, seq, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, seq, 1), minor=True)
        ax.grid(which="minor", color="#888", linewidth=0.3, alpha=0.4)
        ax.tick_params(which="minor", length=0)
        ax.plot([-0.5, seq - 0.5], [-0.5, seq - 0.5], color="#000", linewidth=0.6, alpha=0.5)
        ax.set_xlim(-0.5, seq - 0.5)
        ax.set_ylim(seq - 0.5, -0.5)
        ax.set_xlabel("key token")
        ax.set_ylabel("query token")
        ax.set_title(f"Δ attention · {expanded_title} · α={alpha:+.3g}")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    redraw_key = (split, basis_idx, example_idx, token_idx, float(alpha), selected_cell)

    with rw.VBox() as main:
        rw.HTML(value=style_block)
        with rw.HBox():
            rw.Dropdown(options=split_keys, value=split, on_value=on_split, description="Split:")
            rw.Dropdown(
                options=[(f"cluster {b['cluster_id']} (ex {b['example_id']})", i) for i, b in enumerate(basis_list)],
                value=basis_idx, on_value=set_basis_idx, description="Basis:",
            )
            rw.Dropdown(
                options=[(f"{i}: example {e['example_id']}", i) for i, e in enumerate(current_list)],
                value=example_idx, on_value=on_example, description="Example:",
            )
        _TokenSelector(
            example=tokenized,
            tokenizer=tokenizer,
            selected_idx=token_idx,
            on_select=set_token_idx,
            values=jv_per_token,
            hover_label="J·v",
        )
        with rw.HBox():
            rw.FloatText(value=alpha, on_value=set_alpha, description="α:", layout={"width": "180px"})

        with rw.VBox():
            _MatplotlibView(draw=draw_expanded, redraw_key=redraw_key)

            with rw.VBox(layout={"max_height": "600px", "overflow": "auto", "padding_left": "12px"}):
                with rw.HBox(layout={"flex_flow": "row nowrap"}):
                    _ThumbnailButton(
                        class_name=overall_class,
                        selected=(selected_cell == ("overall",)),
                        tooltip="overall avg",
                        on_click=lambda: set_selected_cell(("overall",)),
                    )
                for L_idx in range(n_down):
                    L = layer + 1 + L_idx
                    with rw.HBox(layout={"flex_flow": "row nowrap"}):
                        _ThumbnailButton(
                            class_name=layer_classes[L_idx],
                            selected=(selected_cell == ("layer", L)),
                            tooltip=f"L{L} avg",
                            on_click=lambda L=L: set_selected_cell(("layer", L)),
                        )
                        for h in range(n_heads):
                            _ThumbnailButton(
                                class_name=head_classes[L_idx][h],
                                selected=(selected_cell == ("head", L, h)),
                                tooltip=f"L{L} H{h}",
                                on_click=lambda L=L, h=h: set_selected_cell(("head", L, h)),
                            )
    return main


def example_attention_delta_widget(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
    alpha_default: float = 1.0,
) -> Element:
    """Pick a (split, example, basis, token, α): show how every downstream attention head's
    pattern shifts under the steering perturbation α·v applied to the layer-`layer` activation
    at the selected token. Requires `model` loaded with `eager_attn=True`."""
    basis_list = _load_basis(cache_path)
    splits: Mapping[str, list[TokenizedExample]] = {
        "probe": list(_load_tok_probe(cache_path)),
        "train": list(_load_tok_train(cache_path)),
    }
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    return _AttentionPatternDelta(
        basis_list=basis_list,
        splits=splits,
        tokenizer=tokenizer,
        model=model,
        layer=layer,
        alpha_default=alpha_default,
    )


def logit_lens_widget(
    cache_path: Path,
    model: LanguageModel,
    top_k: int = 20,
) -> Element:
    basis_list = _load_basis(cache_path)
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]
    W_U: Float[Tensor, "vocab d_model"] = model.lm_head.weight.detach()  # type: ignore[attr-defined]

    def render(basis: BasisDirection) -> Element:
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

        tables_html = (
            f"<div style='display:flex;gap:2em'>"
            f"{fmt_table(top_idx, 'Top upweighted')}"
            f"{fmt_table(bot_idx, 'Top downweighted')}"
            f"</div>"
        )

        def draw() -> None:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.hist(direct_effects, bins=80)
            ax.axvline(0.0, color="gray", linewidth=0.6)
            ax.set_xlabel("direct logit effect  (W_U · v)")
            ax.set_ylabel("token count")
            ax.set_title(f"distribution over vocab (|V|={len(direct_effects)})")
            fig.tight_layout()
            display(fig)
            plt.close(fig)

        with rw.VBox() as box:
            rw.HTML(value=tables_html)
            _MatplotlibView(draw=draw, redraw_key=basis["cluster_id"])
        return box

    return _BasisDropdown(basis_list, render)


def example_cosine_widget(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
) -> Element:
    basis_list = _load_basis(cache_path)
    example_splits = {
        "train": list(_load_tok_train(cache_path)),
        "probe": list(_load_tok_probe(cache_path))
    }
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    def compute(example: TokenizedExample, basis: BasisDirection | None) -> np.ndarray:
        assert basis is not None
        neg_grad = compute_neg_gradients(model, example, layer)
        v = torch.as_tensor(basis["vector"], dtype=neg_grad.dtype, device=neg_grad.device)
        cos = F.cosine_similarity(neg_grad, v.unsqueeze(0), dim=-1)
        return cos.detach().float().cpu().numpy()

    return _ExampleSelector(example_splits, tokenizer, basis_list, compute, "cos")


def example_gradient_norm_widget(
    cache_path: Path,
    model: LanguageModel,
    layer: int = LAYER,
) -> Element:
    example_splits = {
        "train": list(_load_tok_train(cache_path)),
        "probe": list(_load_tok_probe(cache_path))
    }
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(MODEL_ID)  # type: ignore[assignment]

    def compute(example: TokenizedExample, _basis: BasisDirection | None) -> np.ndarray:
        neg_grad = compute_neg_gradients(model, example, layer)
        return neg_grad.norm(dim=-1).detach().float().cpu().numpy()

    return _ExampleSelector(example_splits, tokenizer, None, compute, "‖∇‖")
