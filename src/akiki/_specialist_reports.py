"""Saved specialist evaluations and conservative learning-curve suggestions."""

from dataclasses import dataclass
from enum import Enum
from html import escape
from pathlib import Path

import numpy as np


class TrainingAdvice(Enum):
    """Heuristic next steps based on saved validation scores and loss trends.

    ``CONTINUE_TRAINING`` indicates improving loss or an incomplete trend.
    ``ADD_EXAMPLES`` indicates a flat or worsening recent validation loss.
    ``MORE_VALIDATION_DATA`` indicates fewer than three validation examples
    for an intent, or a missing validation loss.
    ``SUFFICIENT`` indicates strong measured validation scores.
    These suggestions do not establish performance on new inputs.
    """

    CONTINUE_TRAINING = "continue_training"
    ADD_EXAMPLES = "add_examples"
    MORE_VALIDATION_DATA = "more_validation_data"
    SUFFICIENT = "sufficient"


@dataclass(frozen=True)
class IntentEvaluation:
    """Saved validation and training diagnostics for one declared intent.

    ``examples`` counts validation examples; ``training_examples`` counts
    training examples. ``accuracy`` is the validation fraction classified correctly.
    ``cross_entropy`` and ``training_cross_entropy`` are mean losses in nats.
    A metric is ``None`` when the saved report has no corresponding measurement.

    ``affine_gradient_norm`` and ``head_gradient_norm`` describe gradients at
    the selected checkpoint, before clipping. ``gradient_norm`` combines them
    by the square root of the sum of their squares. These are gradients of
    mean per-intent training cross entropy, not validation loss.

    ``loss_slope`` is the fitted change in validation cross entropy per epoch.
    It uses up to four recent saved epochs. It is ``None`` with fewer than two
    measured epochs. ``advice`` is a ``TrainingAdvice`` member; ``reason`` explains it.
    """

    intent: str
    examples: int
    training_examples: int
    accuracy: float | None
    cross_entropy: float | None
    training_cross_entropy: float | None
    gradient_norm: float | None
    affine_gradient_norm: float | None
    head_gradient_norm: float | None
    loss_slope: float | None
    advice: TrainingAdvice
    reason: str


@dataclass(frozen=True)
class SpecialistEvaluation:
    """Held-out scores, selected-state training gradients, and heuristic advice.

    Loss slopes use recent validation epochs. Gradient norms come from the
    gradient of mean per-intent training CE at the selected adapter, before
    clipping. Neither signal guarantees that additional training will help.

    ``per_intent`` maps label names to ``IntentEvaluation`` objects.
    ``accuracy`` is the overall validation fraction classified correctly.
    ``cross_entropy`` is overall validation loss in nats; ``examples`` is
    validation support. No new inference is run to produce this object.

    ``best_epoch`` identifies the checkpoint selected by validation loss.
    It can be zero when continuation keeps its starting checkpoint.
    ``epochs`` counts saved epoch entries for this run, not all earlier runs.
    ``output_dir`` identifies the export containing the report.
    """

    per_intent: dict[str, IntentEvaluation]
    accuracy: float
    cross_entropy: float
    examples: int
    best_epoch: int
    epochs: int
    output_dir: Path

    @property
    def intents_needing_examples(self) -> set[str]:
        """Return intents whose advice is ``TrainingAdvice.ADD_EXAMPLES``.

        Intents needing more validation data are not included in this set.
        """
        return {
            name
            for name, metric in self.per_intent.items()
            if metric.advice is TrainingAdvice.ADD_EXAMPLES
        }

    def _repr_html_(self) -> str:
        def number(value, percentage=False):
            return "—" if value is None else format(value, ".1%" if percentage else ".4g")

        rows = []
        for name, metric in self.per_intent.items():
            cells = [
                escape(name),
                str(metric.examples),
                number(metric.accuracy, True),
                number(metric.cross_entropy),
                number(metric.training_cross_entropy),
                number(metric.affine_gradient_norm),
                number(metric.head_gradient_norm),
                number(metric.loss_slope),
                escape(metric.advice.value),
                escape(metric.reason),
            ]
            rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        headings = (
            "Intent",
            "Validation n",
            "Accuracy",
            "Cross entropy",
            "Training CE",
            "Affine gradient",
            "Head gradient",
            "Validation CE/epoch",
            "Suggestion",
            "Reason",
        )
        return (
            '<section aria-label="Specialist evaluation">'
            f"<p>Validation accuracy: {self.accuracy:.1%}; cross entropy: {self.cross_entropy:.4g}. "
            f"Selected epoch {self.best_epoch}; {self.epochs} epochs observed.</p>"
            "<table><thead><tr>"
            + "".join(f"<th>{heading}</th>" for heading in headings)
            + "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            "<p>Suggestions are heuristics. Gradients describe the selected checkpoint; "
            "loss slopes describe recent validation epochs. More validation data may change the assessment.</p></section>"
        )


def evaluation_from_report(report: dict, output_dir: Path) -> SpecialistEvaluation:
    validation = report["validation"]
    training = report.get("training") or {}
    labels = report["labels"]
    validation_rows = validation.get("per_intent", [])
    training_rows = training.get("per_intent", [])
    if len(validation_rows) != len(labels) or len(training_rows) != len(labels):
        raise ValueError(
            "This export lacks per-intent diagnostics; train it with the updated backend"
        )
    history = report["history"]
    result = {}
    for index, name in enumerate(labels):
        heldout, fitted = validation_rows[index], training_rows[index]
        recent = [
            (epoch["epoch"], epoch["validation"]["per_intent"][index]["nll"])
            for epoch in history[-4:]
            if epoch["validation"].get("per_intent")
            and epoch["validation"]["per_intent"][index]["nll"] is not None
        ]
        slope = None
        if len(recent) >= 2:
            x, y = np.array(recent, dtype=float).T
            slope = float(np.dot(x - x.mean(), y - y.mean()) / np.square(x - x.mean()).sum())
        support, ce, accuracy = heldout["examples"], heldout["nll"], heldout["accuracy"]
        if support < 3 or ce is None:
            advice = TrainingAdvice.MORE_VALIDATION_DATA
            reason = "Too few held-out examples to diagnose a plateau reliably."
        elif accuracy >= 0.95 and ce <= 0.2:
            advice = TrainingAdvice.SUFFICIENT
            reason = "Strong measured held-out scores; confirm coverage before adding training."
        elif slope is None or slope < -max(1e-4, abs(ce) * 1e-3):
            advice = TrainingAdvice.CONTINUE_TRAINING
            reason = (
                "Validation loss is still improving."
                if slope is not None
                else "More epochs are needed to establish a learning trend."
            )
        else:
            advice = TrainingAdvice.ADD_EXAMPLES
            gap = fitted["nll"] is not None and fitted["nll"] + 0.1 < ce
            reason = (
                "Training fits better than validation; add varied examples and check labels."
                if gap
                else "Recent validation loss has flattened or worsened; improve example coverage and inspect errors."
            )
        result[name] = IntentEvaluation(
            name,
            support,
            fitted["examples"],
            accuracy,
            ce,
            fitted["nll"],
            fitted.get("gradient_norm"),
            fitted.get("affine_gradient_norm"),
            fitted.get("head_gradient_norm"),
            slope,
            advice,
            reason,
        )
    return SpecialistEvaluation(
        result,
        validation["accuracy"],
        validation["nll"],
        validation["examples"],
        report["best_epoch"],
        len(history),
        output_dir,
    )
