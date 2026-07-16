#!/usr/bin/env python3
"""
Scalable taxonomy-aware multitask trainer for genomic profile tensors.

Current supervised heads
------------------------
  kingdom  : animalia, plantae, fungi, protista
  organism : 11 current eukaryotic organisms
  element  : 8 universal genomic sites plus 6 human-specific sites

The implementation is intentionally schema-driven around encoders.json and
Y_*.npy label arrays. When prokaryotic/TSS data is added later, the same
pattern can be extended by adding another label array and head definition
instead of rewriting the model.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

DNA_ONEHOT_TABLE = np.zeros((256, 4), dtype=np.float32)
for _base, _idx in ((b"A", 0), (b"C", 1), (b"G", 2), (b"T", 3)):
    DNA_ONEHOT_TABLE[_base[0], _idx] = 1.0
    DNA_ONEHOT_TABLE[_base.lower()[0], _idx] = 1.0

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
existing_tf_xla_flags = os.environ.get("TF_XLA_FLAGS", "")
if "--tf_xla_auto_jit=0" not in existing_tf_xla_flags:
    os.environ["TF_XLA_FLAGS"] = (existing_tf_xla_flags + " --tf_xla_auto_jit=0").strip()


def configure_xla_cuda_data_dir() -> dict[str, str | None | bool]:
    """Point TensorFlow/XLA at CUDA's libdevice directory before TF import."""
    existing_xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_cuda_data_dir=" in existing_xla_flags:
        for token in existing_xla_flags.split():
            if token.startswith("--xla_gpu_cuda_data_dir="):
                cuda_dir = token.split("=", 1)[1]
                libdevice = Path(cuda_dir) / "nvvm" / "libdevice" / "libdevice.10.bc"
                return {
                    "libdevice_found": libdevice.exists(),
                    "libdevice_path": str(libdevice) if libdevice.exists() else None,
                    "xla_gpu_cuda_data_dir": cuda_dir,
                }

    candidates = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDA_ROOT"),
        "/ibdc-hpc/apps1/cuda-12.4",
        "/usr/local/cuda",
        "/usr/lib/cuda",
        "/opt/cuda",
    ]
    for value in candidates:
        if not value:
            continue
        root = Path(value)
        libdevice = root / "nvvm" / "libdevice" / "libdevice.10.bc"
        if libdevice.exists():
            os.environ["XLA_FLAGS"] = (
                existing_xla_flags + f" --xla_gpu_cuda_data_dir={root}"
            ).strip()
            return {
                "libdevice_found": True,
                "libdevice_path": str(libdevice),
                "xla_gpu_cuda_data_dir": str(root),
            }

    return {
        "libdevice_found": False,
        "libdevice_path": None,
        "xla_gpu_cuda_data_dir": None,
    }


RUNTIME_CONFIG = configure_xla_cuda_data_dir()


def ensure_xla_flag(flag: str) -> None:
    existing = os.environ.get("XLA_FLAGS", "")
    if flag not in existing.split():
        os.environ["XLA_FLAGS"] = (existing + f" {flag}").strip()


ensure_xla_flag("--xla_gpu_strict_conv_algorithm_picker=false")
RUNTIME_CONFIG["strict_conv_algorithm_picker"] = False


UNIVERSAL_ELEMENTS = ["es", "ee", "cds", "ge", "gs", "stac", "stoc", "prom"]
HUMAN_SPECIFIC_ELEMENTS = ["3utre", "3utrs", "5utre", "5utrs", "ene", "ens"]
# Merge e/s variants that are indistinguishable from biophysical profiles alone.
HUMAN_SPECIFIC_ELEMENT_MERGE = {
    "3utre": "3utr",
    "3utrs": "3utr",
    "5utre": "5utr",
    "5utrs": "5utr",
    "ene":   "en",
    "ens":   "en",
}
BOUNDARY_COLLAPSE = {
    "es": "exon_boundary",
    "ee": "exon_boundary",
    "gs": "gene_boundary",
    "ge": "gene_boundary",
}
ELEMENT_COARSE = {
    "gs": "gene_boundary",
    "ge": "gene_boundary",
    "gene_boundary": "gene_boundary",
    "prom": "regulatory",
    "5utrs": "utr5",
    "5utre": "utr5",
    "3utrs": "utr3",
    "3utre": "utr3",
    "ens": "intron_boundary",
    "ene": "intron_boundary",
    "es": "exon_boundary",
    "ee": "exon_boundary",
    "exon_boundary": "exon_boundary",
    "cds": "coding",
    "stac": "coding",
    "stoc": "coding",
}
BIOLOGICAL_ELEMENT_GROUPS = [
    ("exon_boundary", ["es", "ee", "exon_boundary"]),
    ("gene_boundary", ["gs", "ge", "gene_boundary"]),
    ("codon_boundary", ["stac", "stoc"]),
    ("cds", ["cds"]),
    ("prom", ["prom"]),
]
PAIR_ELEMENT_HEADS = {
    "pair_es_ee": ["es", "ee"],
    "pair_gs_ge": ["gs", "ge"],
    "pair_stac_stoc": ["stac", "stoc"],
}
HUMAN_SPECIFIC_HEAD = "human_specific_element"
UTR_SUBTYPE_HEAD = "utr_subtype"
CDS_DETECTOR_HEAD = "cds_detector"
UTR_SUBTYPE_MERGE = {
    "3utre": "3utr",
    "3utrs": "3utr",
    "5utre": "5utr",
    "5utrs": "5utr",
}
BOUNDARY_CONTEXT_HEAD = "boundary_context"
BOUNDARY_CONTEXT_MEMBERS = ["gene_boundary", "exon_boundary", "prom"]
START_STOP_PARAMETER_WEIGHTS = {
    "hbond": 2.0,
    "stack": 2.0,
    "sol": 1.4,
    "bp": 1.4,
    "intra": 1.4,
    "bbone": 1.4,
    "inter": 1.0,
}
COARSE_ELEMENT_ORDER = [
    "gene_boundary",
    "regulatory",
    "utr5",
    "utr3",
    "intron_boundary",
    "exon_boundary",
    "coding",
]
# Parameter-specific segment windows, re-derived on the 32-organism cohort (2026-06-19)
# via CESL (Consistent Effect-Size Localization: self-saliency x discriminability x
# organism-consistency, ANOVA->Tukey-Kramer->FDR gate, CDS-relative flatness gate at 2x
# + search-edge guard). None = no localized window for that (kingdom, parameter, element).
# Source: table_windows_rederived/TABLE_WINDOWS_rederived.py  (188 windows, 8 None).
TABLE_SITE_ORDER = ["prom", "gs", "ge", "es", "ee", "stac", "stoc"]
TABLE_WINDOWS = {
    "animalia": {
        "bp": [(248, 277), (208, 242), (203, 237), (223, 252), (223, 252), (248, 282), (198, 237)],
        "inter": [(223, 252), (223, 257), (248, 277), (218, 252), (218, 252), (248, 277), (243, 277)],
        "intra": [(258, 292), (208, 242), (208, 242), (223, 252), (223, 252), (238, 272), (243, 277)],
        "bbone": [(228, 262), (248, 277), (248, 282), (218, 252), (218, 247), (248, 277), (243, 277)],
        "hbond": [(243, 277), (213, 247), (213, 247), (208, 242), (243, 277), (238, 272), (208, 242)],
        "stack": [(188, 222), (243, 277), (238, 272), (228, 262), (228, 262), (233, 267), (223, 257)],
        "sol": [(218, 252), (208, 242), (208, 242), (218, 252), (218, 252), (223, 257), (218, 252)],
    },
    "fungi": {
        "bp": [None, (248, 277), (248, 277), (233, 262), (233, 262), (248, 282), (183, 237)],
        "inter": [(223, 252), (248, 282), (248, 282), (223, 252), (218, 252), (248, 277), (243, 277)],
        "intra": [(243, 277), (253, 282), (253, 282), (223, 252), (223, 252), (238, 272), (243, 277)],
        "bbone": [(203, 237), (223, 257), (248, 282), (248, 277), (248, 277), (223, 257), None],
        "hbond": [(118, 182), (248, 282), None, (253, 287), (253, 287), (243, 277), None],
        "stack": [(218, 252), (238, 272), (238, 267), (218, 252), (218, 252), (228, 262), (208, 237)],
        "sol": [(198, 232), (243, 277), (243, 277), (218, 252), (218, 252), (223, 257), (188, 237)],
    },
    "plantae": {
        "bp": [None, (183, 232), (173, 232), (223, 257), (243, 277), (248, 282), (203, 237)],
        "inter": [None, (223, 257), (223, 257), (243, 277), (218, 252), (248, 277), (218, 252)],
        "intra": [(203, 237), (203, 237), (203, 237), (248, 282), (223, 252), (238, 272), (143, 237)],
        "bbone": [(143, 182), (233, 267), (233, 267), (248, 277), (213, 247), (248, 277), (273, 397)],
        "hbond": [(203, 232), (208, 242), (208, 242), (223, 252), (243, 272), (243, 277), (193, 242)],
        "stack": [None, (148, 212), (168, 212), (243, 277), (243, 277), (233, 267), (148, 272)],
        "sol": [None, (198, 232), (193, 232), (218, 252), (218, 252), (248, 282), (183, 237)],
    },
    "protista": {
        "bp": [(223, 257), (248, 277), (193, 222), (223, 252), (223, 252), (248, 282), (198, 237)],
        "inter": [(223, 257), (223, 257), (223, 257), (218, 252), (218, 252), (248, 277), (243, 277)],
        "intra": [(208, 242), (223, 257), (223, 257), (223, 252), (223, 252), (243, 272), (243, 277)],
        "bbone": [(238, 272), (223, 252), (223, 257), (243, 272), (243, 272), (223, 257), (243, 277)],
        "hbond": [(148, 242), (238, 272), (243, 272), (218, 247), (218, 247), (243, 277), (263, 322)],
        "stack": [(243, 277), (233, 267), (233, 267), (223, 257), (223, 257), (238, 272), (223, 257)],
        "sol": [(223, 257), (223, 257), (223, 257), (243, 277), (243, 277), (248, 282), (208, 242)],
    },
}


def import_tf():
    import tensorflow as tf

    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    return tf


try:
    tf = import_tf()
    SequenceBase = tf.keras.utils.Sequence
except ModuleNotFoundError:
    tf = None

    class SequenceBase:  # type: ignore[override]
        pass


def ordered_labels(mapping: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]


def make_class_weight_vector(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y.astype(np.int32), minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    weights = np.sqrt(len(y) / (n_classes * counts))
    return np.clip(weights, 1.0 / 3.0, 3.0).astype(np.float32)


if tf is not None:
    @tf.keras.utils.register_keras_serializable(package="genome_reader")
    class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
        """Sparse categorical focal loss for hard element classes."""

        def __init__(
            self,
            gamma: float = 2.0,
            from_logits: bool = False,
            label_smoothing: float = 0.0,
            name: str = "sparse_categorical_focal_loss",
            **kwargs,
        ):
            super().__init__(name=name, **kwargs)
            self.gamma = float(gamma)
            self.from_logits = bool(from_logits)
            self.label_smoothing = float(label_smoothing)

        def call(self, y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_pred = tf.convert_to_tensor(y_pred)
            if self.from_logits:
                y_pred = tf.nn.softmax(y_pred, axis=-1)
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0)
            n_classes = tf.shape(y_pred)[-1]
            one_hot = tf.one_hot(y_true, n_classes, dtype=y_pred.dtype)
            # Focal weight uses unsmoothed pt so hard-example focus is preserved.
            pt = tf.reduce_sum(one_hot * y_pred, axis=-1)
            focal_weight = tf.pow(1.0 - pt, self.gamma)
            if self.label_smoothing > 0.0:
                smooth = one_hot * (1.0 - self.label_smoothing) + self.label_smoothing / tf.cast(n_classes, y_pred.dtype)
                cross_entropy = -tf.reduce_sum(smooth * tf.math.log(y_pred), axis=-1)
            else:
                cross_entropy = -tf.math.log(pt)
            return focal_weight * cross_entropy

        def get_config(self):
            config = super().get_config()
            config.update({
                "gamma": self.gamma,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            })
            return config


    @tf.keras.utils.register_keras_serializable(package="genome_reader")
    class MaskedElementAccuracy(tf.keras.metrics.Metric):
        """Accuracy over samples where element sample_weight > 0 only.

        Keras's 'accuracy' string metric in this TF build does NOT apply
        sample_weight to the metric (only to the loss), so zero-weight dummy
        rows (CDS/human-specific) count as wrong predictions and dilute the
        reported accuracy from ~0.72 to ~0.62.  This metric applies the mask
        before computing, producing the honest main-element accuracy.
        """

        def __init__(self, name: str = "masked_accuracy", **kwargs):
            super().__init__(name=name, **kwargs)
            self.numerator   = self.add_weight(name="numerator",   initializer="zeros")
            self.denominator = self.add_weight(name="denominator", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_pred_cls  = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
            correct     = tf.cast(tf.equal(y_true_flat, y_pred_cls), tf.float32)
            if sample_weight is not None:
                mask = tf.cast(tf.reshape(sample_weight, [-1]) > 0, tf.float32)
                self.numerator.assign_add(tf.reduce_sum(correct * mask))
                self.denominator.assign_add(tf.reduce_sum(mask))
            else:
                self.numerator.assign_add(tf.reduce_sum(correct))
                self.denominator.assign_add(tf.cast(tf.size(correct), tf.float32))

        def result(self):
            return tf.math.divide_no_nan(self.numerator, self.denominator)

        def reset_state(self):
            self.numerator.assign(0.0)
            self.denominator.assign(0.0)

        def get_config(self):
            return super().get_config()

    @tf.keras.utils.register_keras_serializable(package="genome_reader")
    class GradientReversal(tf.keras.layers.Layer):
        """Identity forward, negative-scaled gradient backward.

        lambda is held in a non-trainable variable (lambda_var) so it can be ramped
        during training (GRL warmup) without rebuilding the model. With no warmup it
        stays at target_lambda == the old fixed behaviour."""

        def __init__(self, lambda_coeff: float = 1.0, **kwargs):
            super().__init__(**kwargs)
            self.target_lambda = float(lambda_coeff)

        def build(self, input_shape):
            self.lambda_var = self.add_weight(
                name="grl_lambda",
                shape=(),
                initializer=tf.constant_initializer(self.target_lambda),
                trainable=False,
                dtype=tf.float32,
            )
            super().build(input_shape)

        def call(self, inputs):
            lambda_coeff = tf.cast(self.lambda_var, inputs.dtype)

            @tf.custom_gradient
            def _reverse(x):
                def grad(dy):
                    return -lambda_coeff * dy

                return x, grad

            return _reverse(inputs)

        def get_config(self):
            config = super().get_config()
            config.update({"lambda_coeff": self.target_lambda})
            return config


def balance_indices_by_label(
    indices: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    seed: int,
    max_per_class: int | None = None,
    eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Downsample every class to the least represented class count."""
    rng = np.random.default_rng(seed)
    original_indices = np.asarray(indices, dtype=np.int64)
    indices = original_indices
    if eligible_mask is not None:
        eligible = np.asarray(eligible_mask, dtype=np.float32)
        indices = original_indices[eligible[original_indices] > 0]
        if len(indices) == 0:
            raise ValueError("Cannot balance split: no eligible samples remain.")
    y_subset = y[indices].astype(np.int32)

    per_class_indices: list[np.ndarray] = []
    raw_counts: dict[int, int] = {}
    for class_id in range(n_classes):
        class_indices = indices[y_subset == class_id]
        raw_counts[class_id] = int(len(class_indices))
        if len(class_indices) == 0:
            raise ValueError(
                f"Cannot balance split: element class {class_id} has zero samples."
            )
        per_class_indices.append(class_indices)

    balanced_count = min(raw_counts.values())
    if max_per_class is not None and max_per_class > 0:
        balanced_count = min(balanced_count, int(max_per_class))

    sampled_parts = []
    for class_id, class_indices in enumerate(per_class_indices):
        chosen = rng.choice(class_indices, size=balanced_count, replace=False)
        sampled_parts.append(chosen.astype(np.int64))

    balanced = np.concatenate(sampled_parts)
    rng.shuffle(balanced)
    return balanced.astype(np.int64), {
        "balanced_per_element": int(balanced_count),
        "raw_counts": {str(k): int(v) for k, v in raw_counts.items()},
        "balanced_counts": {str(k): int(balanced_count) for k in raw_counts},
        "n_before": int(len(original_indices)),
        "n_eligible": int(len(indices)),
        "n_after": int(len(balanced)),
    }


def balance_indices_by_organism_element(
    indices: np.ndarray,
    y_element: np.ndarray,
    y_organism: np.ndarray,
    n_elements: int,
    n_organisms: int,
    seed: int,
    max_per_combo: int | None = None,
    min_per_combo: int | None = None,
    eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Balance each organism-element combo to a target count.

    With min_per_combo: upsample (with replacement) any combo below the floor.
    With max_per_combo: downsample any combo above the ceiling.
    Both can be combined to define a [min, max] band.
    Without either, every combo is downsampled to the global minimum count
    within that element (legacy behaviour).
    """
    rng = np.random.default_rng(seed)
    original_indices = np.asarray(indices, dtype=np.int64)
    indices = original_indices
    if eligible_mask is not None:
        eligible = np.asarray(eligible_mask, dtype=np.float32)
        indices = original_indices[eligible[original_indices] > 0]
        if len(indices) == 0:
            raise ValueError("Cannot balance split: no eligible organism-element samples remain.")
    y_elem = y_element[indices].astype(np.int32)
    y_org = y_organism[indices].astype(np.int32)

    sampled_parts: list[np.ndarray] = []
    raw_counts: dict[str, int] = {}
    balanced_counts: dict[str, int] = {}
    per_element_targets: dict[str, int] = {}

    use_floor = min_per_combo is not None and min_per_combo > 0
    use_ceil  = max_per_combo is not None and max_per_combo > 0
    legacy_mode = not use_floor and not use_ceil

    for elem_id in range(n_elements):
        elem_mask = y_elem == elem_id
        elem_indices = indices[elem_mask]
        elem_org = y_org[elem_mask]

        per_org: list[np.ndarray] = []
        org_counts: dict[int, int] = {}
        for org_id in range(n_organisms):
            org_elem_indices = elem_indices[elem_org == org_id]
            org_counts[org_id] = int(len(org_elem_indices))
            raw_counts[f"e{elem_id}_o{org_id}"] = org_counts[org_id]
            per_org.append(org_elem_indices)

        present = [c for c in org_counts.values() if c > 0]
        if not present:
            continue

        if legacy_mode:
            legacy_target = min(present)
            per_element_targets[str(elem_id)] = int(legacy_target)

        for org_id, org_elem_indices in enumerate(per_org):
            if len(org_elem_indices) == 0:
                continue
            n = len(org_elem_indices)
            if legacy_mode:
                target = legacy_target
            else:
                target = n
                if use_floor:
                    target = max(target, min_per_combo)
                if use_ceil:
                    target = min(target, max_per_combo)
            replace = target > n
            chosen = rng.choice(org_elem_indices, size=target, replace=replace)
            sampled_parts.append(chosen.astype(np.int64))
            balanced_counts[f"e{elem_id}_o{org_id}"] = int(target)

    if not sampled_parts:
        raise ValueError("No samples remain after organism-element balancing.")

    balanced = np.concatenate(sampled_parts)
    rng.shuffle(balanced)
    selected_counts = list(balanced_counts.values())
    return balanced.astype(np.int64), {
        "requested_max_per_combo": int(max_per_combo) if max_per_combo else None,
        "requested_min_per_combo": int(min_per_combo) if min_per_combo else None,
        "min_balanced_per_combo": int(min(selected_counts)) if selected_counts else 0,
        "max_balanced_per_combo": int(max(selected_counts)) if selected_counts else 0,
        "per_element_targets": per_element_targets,
        "raw_counts": raw_counts,
        "balanced_counts": balanced_counts,
        "n_before": int(len(original_indices)),
        "n_eligible": int(len(indices)),
        "n_after": int(len(balanced)),
    }


def balance_indices_by_kingdom_element(
    indices: np.ndarray,
    y_element: np.ndarray,
    y_kingdom: np.ndarray,
    n_elements: int,
    n_kingdoms: int,
    seed: int,
    max_per_combo: int | None = None,
    eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Downsample so each kingdom contributes comparable counts per element."""
    rng = np.random.default_rng(seed)
    original_indices = np.asarray(indices, dtype=np.int64)
    indices = original_indices
    if eligible_mask is not None:
        eligible = np.asarray(eligible_mask, dtype=np.float32)
        indices = original_indices[eligible[original_indices] > 0]
        if len(indices) == 0:
            raise ValueError("Cannot balance split: no eligible kingdom-element samples remain.")
    y_elem = y_element[indices].astype(np.int32)
    y_king = y_kingdom[indices].astype(np.int32)

    sampled_parts: list[np.ndarray] = []
    raw_counts: dict[str, int] = {}
    balanced_counts: dict[str, int] = {}
    per_element_targets: dict[str, int] = {}

    for elem_id in range(n_elements):
        elem_mask = y_elem == elem_id
        elem_indices = indices[elem_mask]
        elem_king = y_king[elem_mask]

        per_kingdom: list[np.ndarray] = []
        kingdom_counts: dict[int, int] = {}
        for kingdom_id in range(n_kingdoms):
            kingdom_elem_indices = elem_indices[elem_king == kingdom_id]
            kingdom_counts[kingdom_id] = int(len(kingdom_elem_indices))
            raw_counts[f"e{elem_id}_k{kingdom_id}"] = kingdom_counts[kingdom_id]
            per_kingdom.append(kingdom_elem_indices)

        present = [count for count in kingdom_counts.values() if count > 0]
        if not present:
            continue
        target = min(present)
        if max_per_combo is not None and max_per_combo > 0:
            target = min(target, max_per_combo)
        per_element_targets[str(elem_id)] = int(target)

        for kingdom_id, kingdom_elem_indices in enumerate(per_kingdom):
            if len(kingdom_elem_indices) == 0:
                continue
            chosen = rng.choice(kingdom_elem_indices, size=target, replace=False)
            sampled_parts.append(chosen.astype(np.int64))
            balanced_counts[f"e{elem_id}_k{kingdom_id}"] = int(target)

    if not sampled_parts:
        raise ValueError("No samples remain after kingdom-element balancing.")

    balanced = np.concatenate(sampled_parts)
    rng.shuffle(balanced)
    selected_counts = list(balanced_counts.values())
    return balanced.astype(np.int64), {
        "requested_max_per_combo": int(max_per_combo) if max_per_combo else None,
        "min_balanced_per_combo": int(min(selected_counts)) if selected_counts else 0,
        "max_balanced_per_combo": int(max(selected_counts)) if selected_counts else 0,
        "per_element_targets": per_element_targets,
        "raw_counts": raw_counts,
        "balanced_counts": balanced_counts,
        "n_before": int(len(original_indices)),
        "n_eligible": int(len(indices)),
        "n_after": int(len(balanced)),
    }


def append_balanced_masked_head_indices(
    base_indices: np.ndarray,
    source_indices: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    n_classes: int,
    seed: int,
    max_per_class: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Append a separately balanced masked-head subset to a main-head subset."""
    rng = np.random.default_rng(seed)
    base_indices = np.asarray(base_indices, dtype=np.int64)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    masked_indices = source_indices[np.asarray(mask, dtype=np.float32)[source_indices] > 0]
    if len(masked_indices) == 0:
        return base_indices, {
            "enabled": False,
            "reason": "no masked-head samples in source split",
            "n_before": int(len(source_indices)),
            "n_after": int(len(base_indices)),
        }

    balanced_masked, masked_info = balance_indices_by_label(
        masked_indices,
        y,
        n_classes,
        seed=seed,
        max_per_class=max_per_class,
    )
    combined = np.concatenate([base_indices, balanced_masked]).astype(np.int64)
    combined = np.unique(combined)
    rng.shuffle(combined)
    return combined.astype(np.int64), {
        "enabled": True,
        "masked_head_balancing": masked_info,
        "n_main": int(len(base_indices)),
        "n_masked": int(len(balanced_masked)),
        "n_after": int(len(combined)),
    }


def filter_indices_by_scope(
    indices: np.ndarray,
    labels: dict[str, np.ndarray],
    encoders: dict[str, Any],
    scope: str,
    scope_name: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if scope == "global":
        return np.asarray(indices, dtype=np.int64), {"scope": "global", "scope_name": None}
    if not scope_name:
        raise ValueError(f"--scope-name is required when --scope={scope}")

    label_key = {"kingdom": "kingdom", "organism": "organism"}[scope]
    if scope_name not in encoders[label_key]:
        raise ValueError(f"Unknown {scope}: {scope_name}. Known: {sorted(encoders[label_key])}")

    class_id = int(encoders[label_key][scope_name])
    indices = np.asarray(indices, dtype=np.int64)
    filtered = indices[labels[label_key][indices] == class_id]
    if len(filtered) == 0:
        raise ValueError(f"No rows left after filtering {scope}={scope_name}")
    return filtered.astype(np.int64), {
        "scope": scope,
        "scope_name": scope_name,
        "label_key": label_key,
        "class_id": class_id,
        "n_before": int(len(indices)),
        "n_after": int(len(filtered)),
    }


def filter_indices_by_element_set(
    indices: np.ndarray,
    labels: dict[str, np.ndarray],
    encoders: dict[str, Any],
    element_set: str,
    include_human_specific_for_head: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    indices = np.asarray(indices, dtype=np.int64)
    if element_set == "all":
        return indices, {
            "element_set": "all",
            "allowed_elements": "all",
            "n_before": int(len(indices)),
            "n_after": int(len(indices)),
        }

    if element_set != "universal":
        raise ValueError(f"Unknown element set: {element_set}")

    missing = [name for name in UNIVERSAL_ELEMENTS if name not in encoders["element"]]
    if missing:
        raise ValueError(f"Universal element labels missing from encoder: {missing}")

    allowed_names = list(UNIVERSAL_ELEMENTS)
    human_specific_included: list[str] = []
    if include_human_specific_for_head:
        human_specific_included = [
            name for name in HUMAN_SPECIFIC_ELEMENTS
            if name in encoders["element"]
        ]
        allowed_names.extend(human_specific_included)

    allowed_ids = np.asarray(
        [int(encoders["element"][name]) for name in allowed_names],
        dtype=np.int32,
    )
    keep = np.isin(labels["element"][indices], allowed_ids)
    filtered = indices[keep]
    if len(filtered) == 0:
        raise ValueError("No rows left after filtering to universal element classes")
    return filtered.astype(np.int64), {
        "element_set": "universal",
        "allowed_elements": allowed_names,
        "human_specific_included_for_independent_head": human_specific_included,
        "excluded_elements": [
            name for name in HUMAN_SPECIFIC_ELEMENTS
            if name not in human_specific_included
        ],
        "allowed_old_ids": allowed_ids.astype(int).tolist(),
        "n_before": int(len(indices)),
        "n_after": int(len(filtered)),
    }


def remap_labels_to_active_scope(
    labels: dict[str, np.ndarray],
    encoders: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    heads: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], dict[str, Any]]:
    """Remap labels so scoped models only predict classes present in scope."""
    scoped_idx = np.concatenate([train_idx, val_idx, test_idx]).astype(np.int64)
    remapped = dict(labels)
    active_names: dict[str, list[str]] = {}
    remap_info: dict[str, Any] = {}

    for head in heads:
        old_to_name = {int(v): k for k, v in encoders[head].items()}
        active_old_ids = sorted(int(v) for v in np.unique(labels[head][scoped_idx]).tolist())
        mapper = np.full((max(old_to_name) + 1,), -1, dtype=np.int32)
        for new_id, old_id in enumerate(active_old_ids):
            mapper[old_id] = new_id
        remapped[head] = mapper[labels[head].astype(np.int32)]
        active_names[head] = [old_to_name[old_id] for old_id in active_old_ids]
        remap_info[head] = {
            "old_ids": active_old_ids,
            "new_ids": list(range(len(active_old_ids))),
            "labels": active_names[head],
        }

    return remapped, active_names, remap_info


def build_combo_labels(
    first: np.ndarray,
    second: np.ndarray,
    first_names: list[str],
    second_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    first = np.asarray(first, dtype=np.int32)
    second = np.asarray(second, dtype=np.int32)
    valid = (first >= 0) & (second >= 0)
    second_base = int(second.max()) + 1
    packed = first[valid].astype(np.int64) * second_base + second[valid].astype(np.int64)
    unique_packed, inverse = np.unique(packed, return_inverse=True)
    combo = np.full(first.shape, -1, dtype=np.int32)
    combo[valid] = inverse.astype(np.int32)
    pairs = [(int(v // second_base), int(v % second_base)) for v in unique_packed.tolist()]
    labels = [
        {
            "id": i,
            "organism_id": org_id,
            "organism": first_names[org_id],
            "element_id": elem_id,
            "element": second_names[elem_id],
            "label": f"{first_names[org_id]}::{second_names[elem_id]}",
        }
        for i, (org_id, elem_id) in enumerate(pairs)
    ]
    return combo, {"labels": labels, "n_classes": len(labels)}


def build_coarse_element_labels(
    y_element: np.ndarray,
    element_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    coarse_to_id = {name: i for i, name in enumerate(COARSE_ELEMENT_ORDER)}
    element_to_coarse_id = {}
    for element_id, element_name in enumerate(element_names):
        coarse_name = ELEMENT_COARSE.get(element_name, "other")
        if coarse_name not in coarse_to_id:
            coarse_to_id[coarse_name] = len(coarse_to_id)
        element_to_coarse_id[element_id] = coarse_to_id[coarse_name]

    mapper = np.zeros((len(element_names),), dtype=np.int32)
    for element_id, coarse_id in element_to_coarse_id.items():
        mapper[element_id] = coarse_id
    y_element = np.asarray(y_element, dtype=np.int32)
    y_coarse = np.full(y_element.shape, -1, dtype=np.int32)
    valid = y_element >= 0
    y_coarse[valid] = mapper[y_element[valid]]

    labels = [
        {"id": idx, "label": label}
        for label, idx in sorted(coarse_to_id.items(), key=lambda item: item[1])
    ]
    element_map = {
        element_names[element_id]: labels[coarse_id]["label"]
        for element_id, coarse_id in element_to_coarse_id.items()
    }
    return y_coarse.astype(np.int32), {
        "labels": labels,
        "element_to_coarse": element_map,
        "n_classes": len(labels),
    }


def build_biological_group_labels(
    y_element: np.ndarray,
    element_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    group_labels = []
    mapper = np.full((len(element_names),), -1, dtype=np.int32)
    for group_name, members in BIOLOGICAL_ELEMENT_GROUPS:
        active_members = [name for name in members if name in element_names]
        if not active_members:
            continue
        group_id = len(group_labels)
        group_labels.append(
            {
                "id": group_id,
                "label": group_name,
                "elements": active_members,
            }
        )
        for element_name in active_members:
            mapper[element_names.index(element_name)] = group_id

    if not group_labels:
        raise ValueError("No biological element groups are active for this scope")

    y_element = np.asarray(y_element, dtype=np.int32)
    y_group = np.zeros(y_element.shape, dtype=np.int32)
    valid = y_element >= 0
    y_group[valid] = mapper[y_element[valid]]
    if np.any(y_group[valid] < 0):
        missing_ids = sorted(set(y_element[valid][y_group[valid] < 0].astype(int).tolist()))
        missing = [element_names[i] for i in missing_ids]
        raise ValueError(f"Elements are not assigned to a biological group: {missing}")

    return y_group.astype(np.int32), {
        "labels": group_labels,
        "element_to_group": {
            element_name: group_labels[int(mapper[element_id])]["label"]
            for element_id, element_name in enumerate(element_names)
            if mapper[element_id] >= 0
        },
        "n_classes": len(group_labels),
    }


def build_pair_element_labels(
    y_element: np.ndarray,
    element_names: list[str],
    pair_name: str,
    members: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    active_members = [name for name in members if name in element_names]
    if len(active_members) != len(members):
        missing = [name for name in members if name not in element_names]
        raise ValueError(f"Cannot build {pair_name}; missing elements: {missing}")

    member_ids = [element_names.index(name) for name in members]
    y_element = np.asarray(y_element, dtype=np.int32)
    pair_y = np.zeros(y_element.shape, dtype=np.int32)
    pair_mask = np.zeros(y_element.shape, dtype=np.float32)
    for class_id, element_id in enumerate(member_ids):
        mask = y_element == element_id
        pair_y[mask] = class_id
        pair_mask[mask] = 1.0

    return pair_y.astype(np.int32), pair_mask.astype(np.float32), {
        "name": pair_name,
        "labels": [{"id": i, "label": label} for i, label in enumerate(members)],
        "element_ids": member_ids,
        "n_classes": len(members),
    }


def build_subset_element_labels(
    y_element: np.ndarray,
    element_names: list[str],
    head_name: str,
    members: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    active_members = [name for name in members if name in element_names]
    if len(active_members) < 2:
        return (
            np.zeros(y_element.shape, dtype=np.int32),
            np.zeros(y_element.shape, dtype=np.float32),
            {
                "name": head_name,
                "labels": [],
                "element_ids": [],
                "n_classes": 0,
                "enabled": False,
                "reason": "fewer than two requested members are present",
            },
        )

    member_ids = [element_names.index(name) for name in active_members]
    y_element = np.asarray(y_element, dtype=np.int32)
    subset_y = np.zeros(y_element.shape, dtype=np.int32)
    subset_mask = np.zeros(y_element.shape, dtype=np.float32)
    for class_id, element_id in enumerate(member_ids):
        mask = y_element == element_id
        subset_y[mask] = class_id
        subset_mask[mask] = 1.0

    return subset_y.astype(np.int32), subset_mask.astype(np.float32), {
        "name": head_name,
        "labels": [{"id": i, "label": label} for i, label in enumerate(active_members)],
        "element_ids": member_ids,
        "n_classes": len(active_members),
        "enabled": True,
    }


def split_human_specific_element_head(
    y_element: np.ndarray,
    y_organism: np.ndarray,
    element_names: list[str],
    organism_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Keep human-only extra sites out of the universal element head.

    Main element labels are remapped after removing HUMAN_SPECIFIC_ELEMENTS.
    Human-specific labels are trained by a masked independent head. Non-masked
    rows receive a dummy class 0 label and zero sample weight for that head.
    """
    y_element = np.asarray(y_element, dtype=np.int32)
    y_organism = np.asarray(y_organism, dtype=np.int32)
    original_element_names = list(element_names)
    main_element_names = [
        name for name in original_element_names
        if name not in HUMAN_SPECIFIC_ELEMENTS
    ]
    human_specific_names = [
        name for name in HUMAN_SPECIFIC_ELEMENTS
        if name in original_element_names
    ]

    if "human" not in organism_names:
        return (
            y_element.copy(),
            np.zeros(y_element.shape, dtype=np.int32),
            np.ones(y_element.shape, dtype=np.float32),
            np.zeros(y_element.shape, dtype=np.float32),
            main_element_names,
            {
                "enabled": False,
                "reason": "human organism is not active in this scope",
                "requested_labels": HUMAN_SPECIFIC_ELEMENTS,
                "active_labels": human_specific_names,
            },
        )
    if len(human_specific_names) < 2:
        return (
            y_element.copy(),
            np.zeros(y_element.shape, dtype=np.int32),
            np.ones(y_element.shape, dtype=np.float32),
            np.zeros(y_element.shape, dtype=np.float32),
            main_element_names,
            {
                "enabled": False,
                "reason": "fewer than two human-specific element labels are active",
                "requested_labels": HUMAN_SPECIFIC_ELEMENTS,
                "active_labels": human_specific_names,
            },
        )
    if not main_element_names:
        raise ValueError("Cannot create human-specific head without main element labels.")

    main_old_ids = [original_element_names.index(name) for name in main_element_names]
    human_old_ids = [original_element_names.index(name) for name in human_specific_names]
    main_mapper = np.zeros((len(original_element_names),), dtype=np.int32)
    for new_id, old_id in enumerate(main_old_ids):
        main_mapper[old_id] = new_id

    # Build merged label list preserving order of first occurrence.
    merged_label_list: list[str] = []
    for name in human_specific_names:
        merged = HUMAN_SPECIFIC_ELEMENT_MERGE.get(name, name)
        if merged not in merged_label_list:
            merged_label_list.append(merged)
    merged_label_index = {label: i for i, label in enumerate(merged_label_list)}

    human_mapper = np.zeros((len(original_element_names),), dtype=np.int32)
    for old_id, name in zip(human_old_ids, human_specific_names):
        merged = HUMAN_SPECIFIC_ELEMENT_MERGE.get(name, name)
        human_mapper[old_id] = merged_label_index[merged]

    human_organism_id = organism_names.index("human")
    main_mask = np.isin(y_element, np.asarray(main_old_ids, dtype=np.int32))
    human_mask = (
        np.isin(y_element, np.asarray(human_old_ids, dtype=np.int32))
        & (y_organism == human_organism_id)
    )

    main_y = np.zeros(y_element.shape, dtype=np.int32)
    main_y[main_mask] = main_mapper[y_element[main_mask]]
    human_y = np.zeros(y_element.shape, dtype=np.int32)
    human_y[human_mask] = human_mapper[y_element[human_mask]]

    counts = {
        label: int(np.sum(human_y[human_mask] == class_id))
        for class_id, label in enumerate(merged_label_list)
    }
    return (
        main_y.astype(np.int32),
        human_y.astype(np.int32),
        main_mask.astype(np.float32),
        human_mask.astype(np.float32),
        main_element_names,
        {
            "enabled": bool(np.any(human_mask)),
            "name": HUMAN_SPECIFIC_HEAD,
            "labels": [
                {"id": i, "label": label}
                for i, label in enumerate(merged_label_list)
            ],
            "n_classes": len(merged_label_list),
            "human_organism_id": int(human_organism_id),
            "human_organism_label": "human",
            "main_element_labels": main_element_names,
            "removed_from_main_element_head": human_specific_names,
            "sample_counts": counts,
            "n_masked_samples": int(np.sum(human_mask)),
        },
    )


def build_utr_subtype_labels(
    y_element: np.ndarray,
    y_organism: np.ndarray,
    element_names: list[str],
    organism_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """3utr vs 5utr subtype head, masked to human UTR samples only.

    Must be called BEFORE split_human_specific_element_head removes UTR
    elements from labels["element"] and element_names.
    """
    present = {k: v for k, v in UTR_SUBTYPE_MERGE.items() if k in element_names}
    subtypes = list(dict.fromkeys(UTR_SUBTYPE_MERGE.values()))  # ["3utr", "5utr"]

    disabled = (np.zeros_like(y_element), np.zeros(y_element.shape, dtype=np.float32))
    if len(set(present.values())) < 2 or "human" not in organism_names:
        return (*disabled, {"enabled": False, "reason": "insufficient UTR labels or human absent"})

    human_organism_id = organism_names.index("human")
    subtype_index = {s: i for i, s in enumerate(subtypes)}
    utr_old_ids = [element_names.index(name) for name in present]

    mask = (
        np.isin(np.asarray(y_element), np.asarray(utr_old_ids, dtype=np.int32))
        & (np.asarray(y_organism) == human_organism_id)
    )

    mapper = np.zeros(len(element_names), dtype=np.int32)
    for name, subtype in present.items():
        mapper[element_names.index(name)] = subtype_index[subtype]

    utr_y = np.zeros(y_element.shape, dtype=np.int32)
    utr_y[mask] = mapper[y_element[mask]]

    counts = {s: int(np.sum(utr_y[mask] == i)) for i, s in enumerate(subtypes)}
    return (
        utr_y.astype(np.int32),
        mask.astype(np.float32),
        {
            "enabled": True,
            "name": UTR_SUBTYPE_HEAD,
            "labels": [{"id": i, "label": s} for i, s in enumerate(subtypes)],
            "n_classes": 2,
            "members": list(present.keys()),
            "sample_counts": counts,
            "n_masked_samples": int(np.sum(mask)),
        },
    )


def build_cds_detector_labels(
    y_element: np.ndarray,
    element_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Binary CDS vs non-CDS detector trained on all samples.
    Must be called BEFORE split_human_specific_element_head so element IDs
    are still in the original (unsplit) label space.
    """
    if "cds" not in element_names:
        return (
            np.zeros_like(y_element),
            np.zeros(y_element.shape, dtype=np.float32),
            {"enabled": False, "reason": "cds not in element_names"},
        )
    cds_id = element_names.index("cds")
    cds_y = (np.asarray(y_element, dtype=np.int32) == cds_id).astype(np.int32)
    n_cds = int(np.sum(cds_y))
    n_noncds = int(len(cds_y) - n_cds)
    # Class-balanced weights: upweight CDS to counteract imbalance vs non-CDS.
    if n_cds > 0 and n_noncds > 0:
        n_total = n_cds + n_noncds
        w_cds = n_total / (2.0 * n_cds)
        w_noncds = n_total / (2.0 * n_noncds)
        mask = np.where(cds_y == 1, w_cds, w_noncds).astype(np.float32)
    else:
        mask = np.ones(y_element.shape, dtype=np.float32)
    return (
        cds_y,
        mask,
        {
            "enabled": True,
            "name": CDS_DETECTOR_HEAD,
            "labels": [{"id": 0, "label": "non_cds"}, {"id": 1, "label": "cds"}],
            "n_classes": 2,
            "sample_counts": {"cds": n_cds, "non_cds": n_noncds},
            "n_masked_samples": n_cds + n_noncds,
        },
    )


def split_cds_from_element_head(
    y_element: np.ndarray,
    element_names: list[str],
    existing_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Remove CDS from the main element head after split_human_specific_element_head.
    Returns (remapped_labels, new_mask, new_element_names).
    The new_mask zeros out CDS samples so they don't contribute to element loss.
    """
    y_element = np.asarray(y_element, dtype=np.int32)
    if "cds" not in element_names:
        mask = existing_mask if existing_mask is not None else np.ones(y_element.shape, dtype=np.float32)
        return y_element.copy(), mask, list(element_names)

    cds_id = element_names.index("cds")
    new_element_names = [n for n in element_names if n != "cds"]

    old_to_new = np.zeros(len(element_names), dtype=np.int32)
    new_id = 0
    for old_id, name in enumerate(element_names):
        if name != "cds":
            old_to_new[old_id] = new_id
            new_id += 1

    # Guard against dummy label 0 colliding with cds_id when existing_mask is present.
    is_cds = y_element == cds_id
    if existing_mask is not None:
        is_cds = is_cds & (existing_mask > 0)

    new_y = old_to_new[y_element]
    new_y[is_cds] = 0  # dummy for excluded samples

    new_mask = (~is_cds).astype(np.float32)
    if existing_mask is not None:
        new_mask = new_mask * existing_mask

    return new_y.astype(np.int32), new_mask, new_element_names


def collapse_ambiguous_element_labels(
    y_element: np.ndarray,
    element_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    collapsed_names = []
    old_to_new = np.zeros((len(element_names),), dtype=np.int32)
    old_to_label = {}
    for old_id, label in enumerate(element_names):
        new_label = BOUNDARY_COLLAPSE.get(label, label)
        if new_label not in collapsed_names:
            collapsed_names.append(new_label)
        new_id = collapsed_names.index(new_label)
        old_to_new[old_id] = new_id
        old_to_label[label] = new_label

    y_element = np.asarray(y_element, dtype=np.int32)
    y_collapsed = np.full(y_element.shape, -1, dtype=np.int32)
    valid = y_element >= 0
    y_collapsed[valid] = old_to_new[y_element[valid]]
    return y_collapsed.astype(np.int32), collapsed_names, {
        "enabled": True,
        "old_labels": list(element_names),
        "new_labels": collapsed_names,
        "old_to_new_label": old_to_label,
    }


def build_parameter_channel_weights(parameter_to_idx: dict[str, int]) -> np.ndarray:
    weights = np.ones((len(parameter_to_idx),), dtype=np.float32)
    for parameter, weight in START_STOP_PARAMETER_WEIGHTS.items():
        if parameter in parameter_to_idx:
            weights[int(parameter_to_idx[parameter])] = float(weight)
    return weights


def write_history_csv(path: Path, history: dict[str, list[Any]]) -> None:
    keys = list(history.keys())
    n_rows = max((len(v) for v in history.values()), default=0)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for i in range(n_rows):
            writer.writerow([history[k][i] if i < len(history[k]) else "" for k in keys])


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def write_per_class_report(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class_id", "label", "support", "correct", "accuracy"])
        for class_id, label in enumerate(labels):
            mask = y_true == class_id
            support = int(mask.sum())
            correct = int((y_pred[mask] == class_id).sum()) if support else 0
            acc = float(correct / support) if support else 0.0
            writer.writerow([class_id, label, support, correct, acc])


def write_confusion_csv(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> None:
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, pred in zip(y_true.astype(np.int32), y_pred.astype(np.int32)):
        matrix[int(actual), int(pred)] += 1
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def write_kg_posthoc_element_confusions(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    kg_dir: Path,
    min_count: int = 1,
) -> None:
    import pandas as pd

    nodes = pd.read_csv(Path(kg_dir) / "kg_nodes.csv")
    edges = pd.read_csv(Path(kg_dir) / "kg_edges.csv")
    node_rows = {str(row["node_id"]): row for _, row in nodes.iterrows()}
    edge_lookup: dict[tuple[str, str], list[str]] = {}
    for _, row in edges.iterrows():
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        relation = str(row.get("relation", ""))
        edge_lookup.setdefault((source, target), []).append(relation)
        edge_lookup.setdefault((target, source), []).append(relation)

    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, pred in zip(y_true.astype(np.int32), y_pred.astype(np.int32)):
        matrix[int(actual), int(pred)] += 1

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "actual",
            "predicted",
            "count",
            "actual_group",
            "predicted_group",
            "same_group",
            "kg_relations",
            "actual_order",
            "predicted_order",
            "order_distance",
            "actual_n_sequences",
            "predicted_n_sequences",
        ])
        for actual_id, actual_label in enumerate(labels):
            for pred_id, pred_label in enumerate(labels):
                count = int(matrix[actual_id, pred_id])
                if actual_id == pred_id or count < min_count:
                    continue
                actual_node = f"element:{actual_label}"
                pred_node = f"element:{pred_label}"
                actual_group = ELEMENT_COARSE.get(actual_label, "")
                pred_group = ELEMENT_COARSE.get(pred_label, "")
                actual_row = node_rows.get(actual_node, {})
                pred_row = node_rows.get(pred_node, {})
                actual_order = KGFeatureProvider._float_value(actual_row.get("genomic_order", 0.0))
                pred_order = KGFeatureProvider._float_value(pred_row.get("genomic_order", 0.0))
                relations = sorted(set(edge_lookup.get((actual_node, pred_node), [])))
                writer.writerow([
                    actual_label,
                    pred_label,
                    count,
                    actual_group,
                    pred_group,
                    int(actual_group == pred_group),
                    "|".join(relations),
                    actual_order,
                    pred_order,
                    abs(actual_order - pred_order),
                    KGFeatureProvider._float_value(actual_row.get("n_sequences", 0.0)),
                    KGFeatureProvider._float_value(pred_row.get("n_sequences", 0.0)),
                ])


def make_hierarchical_fused_element_predictions(
    element_prob: np.ndarray,
    bio_prob: np.ndarray,
    element_labels: list[str],
    bio_labels: list[str],
    pair_probs: dict[str, np.ndarray],
    pair_schemas: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    fallback_pred = np.argmax(element_prob, axis=1).astype(np.int32)
    bio_pred = np.argmax(bio_prob, axis=1).astype(np.int32)
    fused_pred = fallback_pred.copy()

    element_to_id = {label: idx for idx, label in enumerate(element_labels)}
    direct_group_hits = 0
    codon_pair_hits = 0
    fallback_hits = 0

    pair_pred = None
    pair_labels: list[str] = []
    pair_head = "pair_stac_stoc"
    if pair_head in pair_probs and pair_head in pair_schemas:
        pair_pred = np.argmax(pair_probs[pair_head], axis=1).astype(np.int32)
        pair_labels = [item["label"] for item in pair_schemas[pair_head]["labels"]]

    for i, group_id in enumerate(bio_pred):
        group_label = bio_labels[int(group_id)]
        if group_label in element_to_id:
            fused_pred[i] = element_to_id[group_label]
            direct_group_hits += 1
        elif group_label == "codon_boundary" and pair_pred is not None:
            pair_label = pair_labels[int(pair_pred[i])]
            fused_pred[i] = element_to_id.get(pair_label, fallback_pred[i])
            codon_pair_hits += 1
        else:
            fallback_hits += 1

    return fused_pred.astype(np.int32), {
        "strategy": "bio_element_group_then_pair_stac_stoc_else_element_fallback",
        "direct_group_predictions": int(direct_group_hits),
        "codon_pair_predictions": int(codon_pair_hits),
        "fallback_predictions": int(fallback_hits),
        "requires_heads": ["bio_element_group", "pair_stac_stoc"],
    }


def make_hierarchical_soft_element_probabilities(
    element_prob: np.ndarray,
    bio_prob: np.ndarray,
    element_labels: list[str],
    bio_labels: list[str],
    pair_probs: dict[str, np.ndarray],
    pair_schemas: dict[str, Any],
) -> np.ndarray:
    hierarchy_prob = np.zeros_like(element_prob, dtype=np.float32)
    element_to_id = {label: idx for idx, label in enumerate(element_labels)}

    pair_head = "pair_stac_stoc"
    pair_prob = pair_probs.get(pair_head)
    pair_labels: list[str] = []
    if pair_prob is not None and pair_head in pair_schemas:
        pair_labels = [item["label"] for item in pair_schemas[pair_head]["labels"]]

    for group_id, group_label in enumerate(bio_labels):
        group_mass = bio_prob[:, group_id].astype(np.float32)
        if group_label in element_to_id:
            hierarchy_prob[:, element_to_id[group_label]] += group_mass
        elif group_label == "codon_boundary":
            if pair_prob is not None and pair_labels:
                for pair_id, pair_label in enumerate(pair_labels):
                    element_id = element_to_id.get(pair_label)
                    if element_id is not None:
                        hierarchy_prob[:, element_id] += group_mass * pair_prob[:, pair_id]
            else:
                codon_ids = [element_to_id[name] for name in ("stac", "stoc") if name in element_to_id]
                if codon_ids:
                    split_mass = group_mass / float(len(codon_ids))
                    for element_id in codon_ids:
                        hierarchy_prob[:, element_id] += split_mass

    row_sums = hierarchy_prob.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > 0
    hierarchy_prob[valid] = hierarchy_prob[valid] / row_sums[valid]
    hierarchy_prob[~valid] = element_prob[~valid]
    return hierarchy_prob.astype(np.float32)


def tune_hierarchical_soft_alpha(
    element_prob: np.ndarray,
    hierarchy_prob: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, Any]:
    best_alpha = 0.0
    best_accuracy = -1.0
    curve = []
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = ((1.0 - float(alpha)) * element_prob) + (float(alpha) * hierarchy_prob)
        pred = np.argmax(blended, axis=1).astype(np.int32)
        accuracy = float(np.mean(pred == y_true))
        curve.append({"alpha": round(float(alpha), 2), "accuracy": accuracy})
        if accuracy > best_accuracy:
            best_alpha = float(alpha)
            best_accuracy = accuracy
    return {
        "best_alpha": best_alpha,
        "best_val_accuracy": best_accuracy,
        "curve": curve,
    }


def blend_element_probabilities(
    element_prob: np.ndarray,
    hierarchy_prob: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return ((1.0 - float(alpha)) * element_prob) + (float(alpha) * hierarchy_prob)


def build_table_window_masks(
    parameter_to_idx: dict[str, int],
    n_positions: int,
    original_n_positions: int,
    profile_offset: int,
    table_center: int = 250,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build fixed masks from Supplementary Tables S3/S4 windows.

    Table coordinates are from the paper's 501 bp convention, centered at 250.
    They are shifted into the current profile coordinate system and clipped to
    the model input, which may be a cropped view.
    """
    original_center = original_n_positions // 2
    masks: list[np.ndarray] = []
    counts: list[float] = []
    window_defs: list[dict[str, Any]] = []

    for kingdom, param_windows in TABLE_WINDOWS.items():
        for parameter, site_windows in param_windows.items():
            if parameter not in parameter_to_idx:
                continue
            pidx = int(parameter_to_idx[parameter])
            for element, win in zip(TABLE_SITE_ORDER, site_windows):
                if win is None:          # no informative segment for this (kingdom, param, element)
                    continue
                table_start, table_end = win
                original_start = int(table_start - table_center + original_center)
                original_end = int(table_end - table_center + original_center)
                model_start = max(0, original_start - profile_offset)
                model_end = min(n_positions - 1, original_end - profile_offset)
                if model_end < 0 or model_start >= n_positions or model_start > model_end:
                    continue

                mask = np.zeros((n_positions, len(parameter_to_idx)), dtype=np.float32)
                mask[model_start:model_end + 1, pidx] = 1.0
                mask_count = float(mask.sum())
                if mask_count <= 0:
                    continue

                masks.append(mask)
                counts.append(mask_count)
                window_defs.append(
                    {
                        "kingdom": kingdom,
                        "element": element,
                        "parameter": parameter,
                        "table_start": int(table_start),
                        "table_end": int(table_end),
                        "original_start": int(original_start),
                        "original_end": int(original_end),
                        "model_start": int(model_start),
                        "model_end": int(model_end),
                        "width": int(model_end - model_start + 1),
                    }
                )

    if not masks:
        return (
            np.zeros((0, n_positions, len(parameter_to_idx)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
        )
    return np.stack(masks, axis=0), np.asarray(counts, dtype=np.float32), window_defs


class OrganismProfileNormalizer:
    """Apply train-split per-organism z-score normalization to profile tensors."""

    def __init__(self, means: np.ndarray, stds: np.ndarray, counts: np.ndarray):
        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.asarray(stds, dtype=np.float32)
        self.counts = np.asarray(counts, dtype=np.int64)

    def normalize_raw_batch(self, x_raw_batch: np.ndarray, organism_ids: np.ndarray) -> np.ndarray:
        organism_ids = np.asarray(organism_ids, dtype=np.int32)
        return (
            np.asarray(x_raw_batch, dtype=np.float32) - self.means[organism_ids]
        ) / self.stds[organism_ids]


def compute_organism_profile_normalizer(
    X_raw: np.ndarray,
    organism_labels: np.ndarray,
    train_idx: np.ndarray,
    n_organisms: int,
    eps: float = 1e-6,
    chunk_size: int = 2048,
    heldout_idx: np.ndarray | None = None,
) -> OrganismProfileNormalizer:
    """Estimate per-organism profile mean/std from the active training split.

    Organisms absent from training (e.g. the LOKO held-out kingdom) normally fall
    back to identity (mean 0, std 1). If `heldout_idx` is given, such organisms are
    instead estimated from their OWN profiles in heldout_idx — this is transductive
    INPUT standardization (uses profiles only, never labels → no leak; element
    prediction stays zero-shot). Lets us test whether biophys transfer fails because
    the representation is kingdom-specific vs because held-out orgs got un-normalized
    (raw-scale) input."""

    n_features = int(X_raw.shape[1])
    n_positions = int(X_raw.shape[2])
    sums = np.zeros((n_organisms, n_features, n_positions), dtype=np.float64)
    sum_squares = np.zeros_like(sums)
    counts = np.zeros((n_organisms,), dtype=np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    organism_labels = np.asarray(organism_labels, dtype=np.int32)
    heldout_idx = None if heldout_idx is None else np.asarray(heldout_idx, dtype=np.int64)

    for organism_id in range(n_organisms):
        org_idx = train_idx[organism_labels[train_idx] == organism_id]
        if len(org_idx) == 0 and heldout_idx is not None:
            # organism never seen in training -> estimate input stats from its own
            # held-out profiles (no labels used).
            org_idx = heldout_idx[organism_labels[heldout_idx] == organism_id]
        counts[organism_id] = int(len(org_idx))
        if len(org_idx) == 0:
            continue
        for start in range(0, len(org_idx), chunk_size):
            chunk_idx = np.sort(org_idx[start:start + chunk_size])
            x_chunk = np.asarray(X_raw[chunk_idx], dtype=np.float64)
            sums[organism_id] += x_chunk.sum(axis=0)
            sum_squares[organism_id] += np.square(x_chunk).sum(axis=0)

    means = np.zeros_like(sums, dtype=np.float64)
    stds = np.ones_like(sums, dtype=np.float64)
    for organism_id in range(n_organisms):
        if counts[organism_id] == 0:
            continue
        mean = sums[organism_id] / float(counts[organism_id])
        variance = sum_squares[organism_id] / float(counts[organism_id]) - np.square(mean)
        means[organism_id] = mean
        stds[organism_id] = np.sqrt(np.maximum(variance, eps))
    return OrganismProfileNormalizer(
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
        counts=counts,
    )


def save_organism_profile_normalizer(
    path: Path,
    normalizer: OrganismProfileNormalizer,
    organism_names: list[str],
) -> None:
    np.savez_compressed(
        path,
        means=normalizer.means,
        stds=normalizer.stds,
        counts=normalizer.counts,
        organism_names=np.asarray(organism_names),
    )


def load_organism_profile_normalizer(path: Path) -> OrganismProfileNormalizer:
    data = np.load(path, allow_pickle=True)
    return OrganismProfileNormalizer(
        means=data["means"],
        stds=data["stds"],
        counts=data["counts"],
    )


class MultiHeadProfileSequence(SequenceBase):
    """Memmap-friendly Keras Sequence for several classification heads."""

    def __init__(
        self,
        X_raw: np.ndarray,
        labels: dict[str, np.ndarray],
        indices: np.ndarray,
        batch_size: int,
        class_weights: dict[str, np.ndarray] | None = None,
        sample_weight_masks: dict[str, np.ndarray] | None = None,
        sequence_accessor: "SequenceManifestAccessor | None" = None,
        kg_feature_provider: "KGFeatureProvider | None" = None,
        profile_normalizer: OrganismProfileNormalizer | None = None,
        kingdom_label_conditioning: bool = False,
        n_kingdom_classes: int = 0,
        shuffle: bool = False,
        seed: int = 42,
        output_heads: "set[str] | None" = None,
    ):
        super().__init__()
        self.X_raw = X_raw
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.class_weights = class_weights
        self.sample_weight_masks = sample_weight_masks
        self.sequence_accessor = sequence_accessor
        self.kg_feature_provider = kg_feature_provider
        self.profile_normalizer = profile_normalizer
        self.kingdom_label_conditioning = bool(kingdom_label_conditioning)
        self.n_kingdom_classes = int(n_kingdom_classes)
        self.shuffle = bool(shuffle)
        self.output_heads = output_heads  # if set, only these keys are included in y
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.indices))
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)

    def iteration_indices(self) -> np.ndarray:
        ordered_parts = []
        for i in range(len(self)):  # matches floor-division __len__ — same batches as predict/fit
            start = i * self.batch_size
            ordered_parts.append(np.sort(self.indices[self.order[start:start + self.batch_size]]))
        return np.concatenate(ordered_parts).astype(np.int64) if ordered_parts else np.array([], dtype=np.int64)

    def __getitem__(self, item):
        start = item * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_idx = np.sort(self.indices[self.order[start:end]])

        x_raw = np.asarray(self.X_raw[batch_idx], dtype=np.float32)
        if self.profile_normalizer is not None:
            x_raw = self.profile_normalizer.normalize_raw_batch(
                x_raw,
                self.labels["organism"][batch_idx],
            )
        x_profile = np.transpose(x_raw, (0, 2, 1))
        if self.sequence_accessor is None and self.kg_feature_provider is None and not self.kingdom_label_conditioning:
            x = x_profile
        else:
            x = {"profile": x_profile}
            if self.sequence_accessor is not None:
                x["sequence"] = self.sequence_accessor.read_onehot(batch_idx)
            if self.kg_feature_provider is not None:
                x["kg_features"] = self.kg_feature_provider.features_for(batch_idx)
            if self.kingdom_label_conditioning:
                kingdom_ints = np.asarray(self.labels["kingdom"][batch_idx], dtype=np.int32)
                kingdom_onehot = np.zeros(
                    (len(batch_idx), self.n_kingdom_classes), dtype=np.float32
                )
                kingdom_onehot[np.arange(len(batch_idx)), kingdom_ints] = 1.0
                x["kingdom_label"] = kingdom_onehot
        label_keys = (
            self.output_heads if self.output_heads is not None else self.labels.keys()
        )
        y = {
            head: np.asarray(self.labels[head][batch_idx], dtype=np.int32)
            for head in label_keys
        }

        if self.class_weights is None and self.sample_weight_masks is None:
            return x, y

        sample_weight = {}
        for head in label_keys:
            if self.class_weights is not None and head in self.class_weights:
                weights = self.class_weights[head][y[head]].astype(np.float32)
            else:
                weights = np.ones_like(y[head], dtype=np.float32)
            if self.sample_weight_masks is not None and head in self.sample_weight_masks:
                weights = weights * np.asarray(
                    self.sample_weight_masks[head][batch_idx],
                    dtype=np.float32,
                )
            sample_weight[head] = weights
        return x, y, sample_weight


class SequenceManifestAccessor:
    """Read 501 bp sequences lazily from sequence_manifest.parquet byte offsets."""

    def __init__(
        self,
        manifest_path: Path,
        sequence_root: Path | None = None,
        sequence_length: int = 501,
        preload: bool = False,
    ):
        import pandas as pd

        self.manifest_path = Path(manifest_path)
        self.sequence_root = Path(sequence_root) if sequence_root is not None else self.manifest_path.parent
        self.sequence_length = int(sequence_length)
        columns = ["global_row", "seq_path", "seq_byte_start", "seq_byte_end", "seq_length"]
        df = pd.read_parquet(self.manifest_path, columns=columns)
        max_row = int(df["global_row"].max())
        self.path_ids = np.full((max_row + 1,), -1, dtype=np.int32)
        self.starts = np.full((max_row + 1,), -1, dtype=np.int64)
        self.ends = np.full((max_row + 1,), -1, dtype=np.int64)
        paths, path_ids = np.unique(df["seq_path"].astype(str).to_numpy(), return_inverse=True)
        rows = df["global_row"].to_numpy(dtype=np.int64)
        self.path_ids[rows] = path_ids.astype(np.int32)
        self.starts[rows] = df["seq_byte_start"].to_numpy(dtype=np.int64)
        self.ends[rows] = df["seq_byte_end"].to_numpy(dtype=np.int64)
        self.paths = [self._resolve_path(path) for path in paths.tolist()]
        self.handles: dict[int, Any] = {}
        self._cache: np.ndarray | None = None
        if preload:
            self._preload_all()

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self.sequence_root / p

    def _handle(self, path_id: int):
        if path_id not in self.handles:
            self.handles[path_id] = open(self.paths[path_id], "rb")
        return self.handles[path_id]

    def _preload_all(self) -> None:
        """Load all sequences into RAM as uint8 one-hot. Eliminates per-batch file I/O."""
        n = len(self.path_ids)
        mem_gb = n * self.sequence_length * 4 / 1e9
        print(f"Pre-loading {n:,} sequences into RAM (~{mem_gb:.1f} GB uint8)...")
        cache = np.zeros((n, self.sequence_length, 4), dtype=np.uint8)
        valid_rows = np.where(self.path_ids >= 0)[0]
        # Sort by (path_id, byte_start) for sequential reads — much faster on network FS
        sort_order = np.lexsort((self.starts[valid_rows], self.path_ids[valid_rows]))
        for global_row in valid_rows[sort_order]:
            path_id = int(self.path_ids[global_row])
            start = int(self.starts[global_row])
            end = int(self.ends[global_row])
            if end <= start:
                continue
            fh = self._handle(path_id)
            fh.seek(start)
            raw = fh.read(end - start).strip()[: self.sequence_length]
            if not raw:
                continue
            arr = np.frombuffer(raw, dtype=np.uint8)
            encoded = DNA_ONEHOT_TABLE[arr]
            cache[global_row, : min(len(encoded), self.sequence_length)] = encoded[: self.sequence_length]
        for fh in self.handles.values():
            fh.close()
        self.handles = {}
        self._cache = cache
        print(f"Sequence pre-load complete. Shape: {cache.shape}, dtype: {cache.dtype}")

    def read_onehot(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if self._cache is not None:
            return self._cache[indices].astype(np.float32)
        out = np.zeros((len(indices), self.sequence_length, 4), dtype=np.float32)
        for row_i, global_row in enumerate(indices.tolist()):
            path_id = int(self.path_ids[global_row])
            start = int(self.starts[global_row])
            end = int(self.ends[global_row])
            if path_id < 0 or start < 0 or end <= start:
                continue
            fh = self._handle(path_id)
            fh.seek(start)
            raw = fh.read(end - start).strip()[: self.sequence_length]
            if not raw:
                continue
            arr = np.frombuffer(raw, dtype=np.uint8)
            encoded = DNA_ONEHOT_TABLE[arr]
            out[row_i, : min(len(encoded), self.sequence_length), :] = encoded[: self.sequence_length]
        return out


class KGFeatureProvider:
    """Batch KG-derived node features for non-target context labels.

    Do not include the true element node as an input feature for element
    classification; that would leak the answer into the model. Element KG nodes
    are used only in posthoc reports unless a label-prototype classifier is
    added explicitly.
    """

    def __init__(
        self,
        kg_dir: Path,
        labels: dict[str, np.ndarray],
        active_label_names: dict[str, list[str]],
    ):
        import pandas as pd

        self.labels = labels
        self.active_label_names = active_label_names
        kg_dir = Path(kg_dir)
        nodes_path = kg_dir / "kg_nodes.csv"
        edges_path = kg_dir / "kg_edges.csv"
        if not nodes_path.exists() or not edges_path.exists():
            raise FileNotFoundError(f"KG files not found under {kg_dir}")

        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)
        degree: dict[str, int] = {}
        relation_degree: dict[tuple[str, str], int] = {}
        for _, row in edges.iterrows():
            source = str(row.get("source", ""))
            target = str(row.get("target", ""))
            relation = str(row.get("relation", ""))
            for node_id in (source, target):
                if not node_id:
                    continue
                degree[node_id] = degree.get(node_id, 0) + 1
                key = (node_id, relation)
                relation_degree[key] = relation_degree.get(key, 0) + 1

        node_rows = {str(row["node_id"]): row for _, row in nodes.iterrows()}
        self.feature_tables = {
            "kingdom": self._build_table("kingdom", node_rows, degree, relation_degree),
            "organism": self._build_table("organism", node_rows, degree, relation_degree),
        }
        self.feature_dim = int(
            sum(table.shape[1] for table in self.feature_tables.values())
        )
        print("KGFeatureProvider columns per node:")
        print("  REMOVED (identity): kingdom_idx, organism_idx")
        print("  KEPT structural (10): degree | belongs_to | has_element | "
              "precedes | has_profile | element_idx | genomic_order | "
              "n_organisms | n_elements | n_sequences")
        print("  KEPT profile summary (28): mean/std/peak_pos/peak_val "
              "x bbone bp hbond inter intra sol stack")
        print(f"  feature_dim = {self.feature_dim}  "
              f"({self.feature_tables['organism'].shape[1]} organism "
              f"+ {self.feature_tables['kingdom'].shape[1]} kingdom)")

    @staticmethod
    def _float_value(value: Any) -> float:
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    def _node_features(
        self,
        node_id: str,
        node_rows: dict[str, Any],
        degree: dict[str, int],
        relation_degree: dict[tuple[str, str], int],
    ) -> list[float]:
        row = node_rows.get(node_id)
        if row is None:
            return [0.0] * 38   # 10 structural + 28 profile_summary

        features = [
            np.log1p(float(degree.get(node_id, 0))),
            np.log1p(float(relation_degree.get((node_id, "belongs_to"), 0))),
            np.log1p(float(relation_degree.get((node_id, "has_element"), 0))),
            np.log1p(float(relation_degree.get((node_id, "precedes"), 0))),
            np.log1p(float(relation_degree.get((node_id, "has_profile"), 0))),
            self._float_value(row.get("element_idx", 0.0)),
            self._float_value(row.get("genomic_order", 0.0)),
            np.log1p(max(self._float_value(row.get("n_organisms", 0.0)), 0.0)),
            np.log1p(max(self._float_value(row.get("n_elements", 0.0)), 0.0)),
            np.log1p(max(self._float_value(row.get("n_sequences", 0.0)), 0.0)),
        ]

        profile_summary = row.get("profile_summary", "")
        profile_values: list[float] = []
        if isinstance(profile_summary, str) and profile_summary.strip():
            try:
                summary = json.loads(profile_summary)
                for param in ("bbone", "bp", "hbond", "inter", "intra", "sol", "stack"):
                    part = summary.get(param, {})
                    profile_values.extend([
                        self._float_value(part.get("mean", 0.0)),
                        self._float_value(part.get("std", 0.0)),
                        self._float_value(part.get("peak_pos", 0.0)) / 475.0,
                        self._float_value(part.get("peak_val", 0.0)),
                    ])
            except Exception:
                profile_values = []
        if len(profile_values) < 28:
            profile_values.extend([0.0] * (28 - len(profile_values)))
        return features + profile_values[:28]

    def _build_table(
        self,
        head: str,
        node_rows: dict[str, Any],
        degree: dict[str, int],
        relation_degree: dict[tuple[str, str], int],
    ) -> np.ndarray:
        rows = [
            self._node_features(f"{head}:{name}", node_rows, degree, relation_degree)
            for name in self.active_label_names[head]
        ]
        table = np.asarray(rows, dtype=np.float32)
        scale = np.maximum(np.nanstd(table, axis=0, keepdims=True), 1e-6)
        center = np.nanmean(table, axis=0, keepdims=True)
        return np.nan_to_num((table - center) / scale).astype(np.float32)

    def features_for(self, indices: np.ndarray) -> np.ndarray:
        parts = []
        for head in ("kingdom", "organism"):
            label_ids = self.labels[head][indices].astype(np.int32)
            parts.append(self.feature_tables[head][label_ids])
        return np.concatenate(parts, axis=1).astype(np.float32)


if tf is not None:
    @tf.keras.utils.register_keras_serializable(package="GenomeReader")
    class ParameterScale(tf.keras.layers.Layer):
        """Paper-guided fixed channel scaling for distinctive start/stop parameters."""

        def __init__(self, weights, **kwargs):
            super().__init__(**kwargs)
            self.weights_init = np.asarray(weights, dtype=np.float32)

        def build(self, input_shape):
            self.channel_weights = self.add_weight(
                name="channel_weights",
                shape=self.weights_init.shape,
                initializer=tf.keras.initializers.Constant(self.weights_init),
                trainable=False,
            )

        def call(self, inputs):
            return inputs * tf.reshape(self.channel_weights, (1, 1, -1))

        def get_config(self):
            config = super().get_config()
            config.update({"weights": self.weights_init.tolist()})
            return config


    @tf.keras.utils.register_keras_serializable(package="GenomeReader")
    class StartStopContrastFeatures(tf.keras.layers.Layer):
        """Focused statistics for start/stop codon parameters from the paper.

        Uses H-bond and stacking as primary signals and includes solvation,
        BP-axis, intra, and backbone shifts as supporting structural signals.
        """

        def __init__(self, parameter_indices, half_width: int = 25, **kwargs):
            super().__init__(**kwargs)
            self.parameter_indices_init = [int(v) for v in parameter_indices]
            self.half_width = int(half_width)

        def call(self, inputs):
            x = tf.gather(inputs, self.parameter_indices_init, axis=2)
            n_pos = tf.shape(x)[1]
            center = n_pos // 2
            left_start = tf.maximum(center - (2 * self.half_width), 0)
            left_end = tf.maximum(center - 1, 0)
            core_start = tf.maximum(center - self.half_width, 0)
            core_end = tf.minimum(center + self.half_width + 1, n_pos)
            right_start = tf.minimum(center + 1, n_pos)
            right_end = tf.minimum(center + (2 * self.half_width) + 1, n_pos)

            left = tf.reduce_mean(x[:, left_start:left_end, :], axis=1)
            core = tf.reduce_mean(x[:, core_start:core_end, :], axis=1)
            right = tf.reduce_mean(x[:, right_start:right_end, :], axis=1)
            return tf.concat([left, core, right, right - left, core - left, right - core], axis=1)

        def get_config(self):
            config = super().get_config()
            config.update({
                "parameter_indices": self.parameter_indices_init,
                "half_width": self.half_width,
            })
            return config


    @tf.keras.utils.register_keras_serializable(package="GenomeReader")
    class TableWindowMean(tf.keras.layers.Layer):
        """Mean-pool fixed parameter/position windows from Tables S3/S4."""

        def __init__(self, masks, counts, **kwargs):
            super().__init__(**kwargs)
            self.masks_init = np.asarray(masks, dtype=np.float32)
            self.counts_init = np.asarray(counts, dtype=np.float32)

        def build(self, input_shape):
            self.masks = self.add_weight(
                name="masks",
                shape=self.masks_init.shape,
                initializer=tf.keras.initializers.Constant(self.masks_init),
                trainable=False,
            )
            self.counts = self.add_weight(
                name="counts",
                shape=self.counts_init.shape,
                initializer=tf.keras.initializers.Constant(self.counts_init),
                trainable=False,
            )

        def call(self, inputs):
            pooled = tf.einsum("blf,wlf->bw", inputs, self.masks)
            return pooled / tf.maximum(tf.reshape(self.counts, (1, -1)), 1.0)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "masks": self.masks_init.tolist(),
                    "counts": self.counts_init.tolist(),
                }
            )
            return config


    @tf.keras.utils.register_keras_serializable(package="GenomeReader")
    class ProfileComplexityFeatures(tf.keras.layers.Layer):
        """Entropy, moment, and cumulative Fourier power features per channel."""

        def call(self, inputs):
            x = tf.cast(inputs, tf.float32)  # (batch, position, parameter)
            eps = tf.constant(1e-6, dtype=tf.float32)

            mean = tf.reduce_mean(x, axis=1)
            centered = x - tf.expand_dims(mean, axis=1)
            std = tf.math.reduce_std(x, axis=1)
            std_safe = tf.maximum(std, eps)
            z = centered / tf.expand_dims(std_safe, axis=1)
            skew = tf.reduce_mean(tf.pow(z, 3.0), axis=1)
            kurt = tf.reduce_mean(tf.pow(z, 4.0), axis=1) - 3.0
            xmin = tf.reduce_min(x, axis=1)
            xmax = tf.reduce_max(x, axis=1)
            dyn_range = xmax - xmin

            mass = tf.abs(x) + eps
            prob = mass / tf.reduce_sum(mass, axis=1, keepdims=True)
            entropy = -tf.reduce_sum(prob * tf.math.log(prob), axis=1)
            entropy = entropy / tf.math.log(tf.cast(tf.shape(x)[1], tf.float32))

            xf = tf.transpose(x, [0, 2, 1])  # (batch, parameter, position)
            fft = tf.signal.rfft(xf)
            power = tf.square(tf.abs(fft))
            power = power[:, :, 1:]  # drop DC term
            power_sum = tf.reduce_sum(power, axis=2, keepdims=True)
            power_prob = power / tf.maximum(power_sum, eps)
            freq = tf.cast(tf.range(tf.shape(power)[2]), tf.float32)
            freq = freq / tf.maximum(tf.cast(tf.shape(power)[2] - 1, tf.float32), 1.0)
            freq = tf.reshape(freq, (1, 1, -1))
            spectral_mean = tf.reduce_sum(power_prob * freq, axis=2)
            spectral_centered = freq - tf.expand_dims(spectral_mean, axis=2)
            spectral_var = tf.reduce_sum(power_prob * tf.square(spectral_centered), axis=2)
            spectral_std = tf.sqrt(tf.maximum(spectral_var, eps))

            cfs = tf.cumsum(power_prob, axis=2)
            cfs_mean = tf.reduce_mean(cfs, axis=2)
            cfs_centered = cfs - tf.expand_dims(cfs_mean, axis=2)
            cfs_std = tf.math.reduce_std(cfs, axis=2)
            cfs_std_safe = tf.maximum(cfs_std, eps)
            cfs_skew = tf.reduce_mean(
                tf.pow(cfs_centered / tf.expand_dims(cfs_std_safe, axis=2), 3.0),
                axis=2,
            )
            cfs_kurt = tf.reduce_mean(
                tf.pow(cfs_centered / tf.expand_dims(cfs_std_safe, axis=2), 4.0),
                axis=2,
            ) - 3.0

            features = [
                mean, std, skew, kurt, xmin, xmax, dyn_range, entropy,
                spectral_mean, spectral_std, cfs_mean, cfs_std, cfs_skew, cfs_kurt,
            ]
            return tf.concat(features, axis=1)


    @tf.keras.utils.register_keras_serializable(package="GenomeReader")
    class ProfileCapsulePooling(tf.keras.layers.Layer):
        """Summarize profile feature maps as position-aware structural capsules."""

        def __init__(self, num_capsules: int = 16, capsule_dim: int = 8, **kwargs):
            super().__init__(**kwargs)
            self.num_capsules = int(num_capsules)
            self.capsule_dim = int(capsule_dim)

        @staticmethod
        def squash(vectors):
            squared_norm = tf.reduce_sum(tf.square(vectors), axis=-1, keepdims=True)
            scale = squared_norm / (1.0 + squared_norm)
            return scale * vectors / tf.sqrt(squared_norm + 1e-7)

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            n_positions = tf.shape(inputs)[1]
            caps = tf.reshape(
                inputs,
                (batch_size, n_positions, self.num_capsules, self.capsule_dim),
            )
            caps = self.squash(caps)
            activations = tf.norm(caps, axis=-1)
            position_weights = tf.nn.softmax(activations, axis=1)
            summary = tf.reduce_sum(caps * tf.expand_dims(position_weights, axis=-1), axis=1)
            lengths = tf.norm(summary, axis=-1)
            output = tf.concat([tf.reshape(summary, (batch_size, -1)), lengths], axis=1)
            output.set_shape((None, self.num_capsules * self.capsule_dim + self.num_capsules))
            return output

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.num_capsules * self.capsule_dim + self.num_capsules)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "num_capsules": self.num_capsules,
                    "capsule_dim": self.capsule_dim,
                }
            )
            return config


    @tf.keras.utils.register_keras_serializable(package="GenomeReader")
    class ExplicitMotifFeatures(tf.keras.layers.Layer):
        """Position-aware exact motif and composition features from one-hot DNA."""

        DEFAULT_MOTIFS = [
            ("start_atg", "ATG"),
            ("stop_taa", "TAA"),
            ("stop_tag", "TAG"),
            ("stop_tga", "TGA"),
            ("splice_donor_gt", "GT"),
            ("splice_acceptor_ag", "AG"),
            ("tata_core", "TATA"),
            ("tata_box", "TATAAA"),
            ("caat_box", "CAAT"),
            ("gc_box", "GGGCGG"),
            ("cpg", "CG"),
            ("poly_a_signal", "AATAAA"),
        ]

        BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}

        def __init__(
            self,
            sequence_length: int = 501,
            center: int | None = None,
            center_width: int = 18,
            flank_width: int = 80,
            motifs: list[tuple[str, str]] | None = None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.sequence_length = int(sequence_length)
            self.center = int(center) if center is not None else int(sequence_length) // 2
            self.center_width = int(center_width)
            self.flank_width = int(flank_width)
            self.motifs = [(str(name), str(pattern).upper()) for name, pattern in (motifs or self.DEFAULT_MOTIFS)]
            self.lengths = sorted({len(pattern) for _, pattern in self.motifs})
            self.kernel_weights = []
            self.window_slices: dict[int, dict[str, tuple[int, int]]] = {}

        def build(self, input_shape):
            self.kernel_weights = []
            self.window_slices = {}
            for motif_length in self.lengths:
                group = [(name, pattern) for name, pattern in self.motifs if len(pattern) == motif_length]
                kernel = np.zeros((motif_length, 4, len(group)), dtype=np.float32)
                for motif_id, (_, pattern) in enumerate(group):
                    for position, base in enumerate(pattern):
                        kernel[position, self.BASE_TO_INDEX[base], motif_id] = 1.0
                self.kernel_weights.append(
                    self.add_weight(
                        name=f"explicit_motif_kernel_{motif_length}",
                        shape=kernel.shape,
                        initializer=tf.keras.initializers.Constant(kernel),
                        trainable=False,
                    )
                )

                valid_length = max(self.sequence_length - motif_length + 1, 1)
                center = min(max(self.center, 0), valid_length - 1)
                self.window_slices[motif_length] = {
                    "center": (
                        max(0, center - self.center_width),
                        min(valid_length, center + self.center_width + 1),
                    ),
                    "upstream": (
                        max(0, center - self.flank_width),
                        max(1, center),
                    ),
                    "downstream": (
                        min(valid_length - 1, center + 1),
                        min(valid_length, center + self.flank_width + 1),
                    ),
                }

        @staticmethod
        def _slice_mean(x, start: int, end: int):
            if end <= start:
                end = start + 1
            return tf.reduce_mean(x[:, start:end, :], axis=1)

        @staticmethod
        def _slice_max(x, start: int, end: int):
            if end <= start:
                end = start + 1
            return tf.reduce_max(x[:, start:end, :], axis=1)

        def call(self, inputs):
            seq = tf.cast(inputs, tf.float32)
            features = []
            for motif_length, kernel in zip(self.lengths, self.kernel_weights):
                score = tf.nn.conv1d(seq, kernel, stride=1, padding="VALID")
                score = tf.clip_by_value(score / float(motif_length), 0.0, 1.0)
                windows = self.window_slices[motif_length]
                upstream = self._slice_mean(score, *windows["upstream"])
                downstream = self._slice_mean(score, *windows["downstream"])
                features.extend(
                    [
                        tf.reduce_max(score, axis=1),
                        tf.reduce_mean(score, axis=1),
                        self._slice_max(score, *windows["center"]),
                        upstream,
                        downstream,
                        downstream - upstream,
                    ]
                )

            base_center_start = max(0, self.center - self.center_width)
            base_center_end = min(self.sequence_length, self.center + self.center_width + 1)
            base_upstream_start = max(0, self.center - self.flank_width)
            base_upstream_end = max(1, self.center)
            base_downstream_start = min(self.sequence_length - 1, self.center + 1)
            base_downstream_end = min(self.sequence_length, self.center + self.flank_width + 1)

            center_base = self._slice_mean(seq, base_center_start, base_center_end)
            upstream_base = self._slice_mean(seq, base_upstream_start, base_upstream_end)
            downstream_base = self._slice_mean(seq, base_downstream_start, base_downstream_end)
            base_asymmetry = downstream_base - upstream_base

            center_gc = center_base[:, 1:2] + center_base[:, 2:3]
            upstream_gc = upstream_base[:, 1:2] + upstream_base[:, 2:3]
            downstream_gc = downstream_base[:, 1:2] + downstream_base[:, 2:3]
            center_pyr = center_base[:, 1:2] + center_base[:, 3:4]
            upstream_pyr = upstream_base[:, 1:2] + upstream_base[:, 3:4]
            downstream_pyr = downstream_base[:, 1:2] + downstream_base[:, 3:4]
            composition = tf.concat(
                [
                    center_base,
                    upstream_base,
                    downstream_base,
                    base_asymmetry,
                    center_gc,
                    upstream_gc,
                    downstream_gc,
                    downstream_gc - upstream_gc,
                    center_pyr,
                    upstream_pyr,
                    downstream_pyr,
                    downstream_pyr - upstream_pyr,
                ],
                axis=1,
            )
            features.append(composition)
            return tf.concat(features, axis=1)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "sequence_length": self.sequence_length,
                    "center": self.center,
                    "center_width": self.center_width,
                    "flank_width": self.flank_width,
                    "motifs": self.motifs,
                }
            )
            return config


def convnext_block(x, d_model: int, kernel_size: int, dropout: float):
    residual = x
    x = tf.keras.layers.Conv1D(
        d_model,
        kernel_size,
        padding="same",
        groups=d_model,
        use_bias=False,
    )(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dense(d_model * 4, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(d_model)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return tf.keras.layers.Add()([residual, x])


def build_model(
    n_positions: int,
    n_features: int,
    head_classes: dict[str, int],
    d_model: int,
    n_stages: int,
    blocks_per_stage: int,
    stem_kernel: int,
    block_kernel: int,
    dropout: float,
    table_window_masks: np.ndarray | None = None,
    table_window_counts: np.ndarray | None = None,
    use_table_windows: bool = True,
    use_complexity_features: bool = True,
    use_profile_capsule_branch: bool = False,
    profile_capsule_num_capsules: int = 16,
    profile_capsule_dim: int = 8,
    use_sequence_branch: bool = False,
    use_motif_detection_branch: bool = False,
    motif_detection_dim: int = 32,
    sequence_length: int = 501,
    use_kg_branch: bool = False,
    kg_feature_dim: int = 0,
    legacy_kingdom_conditioning: bool = False,
    kingdom_label_conditioning: bool = False,
    kingdom_gradient_reversal_lambda: float = 0.0,
    condition_element_on_organism: bool = True,
    organism_gradient_reversal_lambda: float = 0.0,
    parameter_channel_weights: np.ndarray | None = None,
    start_stop_parameter_indices: list[int] | None = None,
    use_start_stop_contrast: bool = False,
    kingdom_specific_organism_head: bool = False,
    kingdom_organism_head_classes: "dict[str, int] | None" = None,
    kingdom_specific_element_head: bool = False,
    kingdom_element_head_classes: "dict[str, int] | None" = None,
):
    keras = tf.keras
    profile_inputs = keras.layers.Input(shape=(n_positions, n_features), name="profile")
    profile_x = profile_inputs
    if parameter_channel_weights is not None:
        profile_x = ParameterScale(parameter_channel_weights, name="paper_parameter_scale")(profile_x)

    x = keras.layers.Conv1D(d_model, stem_kernel, padding="same", name="stem")(profile_x)
    for stage in range(n_stages):
        for _ in range(blocks_per_stage):
            x = convnext_block(x, d_model, block_kernel, dropout)
        if stage < n_stages - 1:
            x = keras.layers.Conv1D(d_model, 2, strides=2, padding="same")(x)

    profile_feature_map = x
    x = keras.layers.GlobalAveragePooling1D(name="shared_pool")(x)
    x = keras.layers.Dense(256, activation="relu", name="shared_dense_1")(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(128, activation="relu", name="shared_dense_2")(x)
    x = keras.layers.Dropout(dropout)(x)

    if use_profile_capsule_branch:
        n_capsules = max(2, int(profile_capsule_num_capsules))
        capsule_dim = max(2, int(profile_capsule_dim))
        cap_x = keras.layers.Conv1D(
            n_capsules * capsule_dim,
            9,
            padding="same",
            name="profile_capsule_votes",
        )(profile_feature_map)
        cap_x = ProfileCapsulePooling(
            num_capsules=n_capsules,
            capsule_dim=capsule_dim,
            name="profile_capsule_pooling",
        )(cap_x)
        cap_x = keras.layers.LayerNormalization(epsilon=1e-6, name="profile_capsule_norm")(cap_x)
        cap_x = keras.layers.Dense(96, activation="relu", name="profile_capsule_dense_1")(cap_x)
        cap_x = keras.layers.Dropout(dropout)(cap_x)
        cap_x = keras.layers.Dense(48, activation="relu", name="profile_capsule_dense_2")(cap_x)
        x = keras.layers.Concatenate(name="shared_plus_profile_capsules")([x, cap_x])
        x = keras.layers.Dense(176, activation="relu", name="profile_capsule_fusion_dense")(x)
        x = keras.layers.Dropout(dropout)(x)

    if (
        use_table_windows
        and table_window_masks is not None
        and table_window_counts is not None
        and len(table_window_masks) > 0
    ):
        window_x = TableWindowMean(
            table_window_masks,
            table_window_counts,
            name="table_window_means",
        )(profile_x)
        window_x = keras.layers.LayerNormalization(epsilon=1e-6, name="table_window_norm")(window_x)
        window_x = keras.layers.Dense(128, activation="relu", name="table_window_dense_1")(window_x)
        window_x = keras.layers.Dropout(dropout)(window_x)
        window_x = keras.layers.Dense(64, activation="relu", name="table_window_dense_2")(window_x)
        x = keras.layers.Concatenate(name="shared_plus_table_windows")([x, window_x])
        x = keras.layers.Dense(160, activation="relu", name="fusion_dense")(x)
        x = keras.layers.Dropout(dropout)(x)

    if use_complexity_features:
        complexity_x = ProfileComplexityFeatures(name="profile_complexity_features")(profile_x)
        complexity_x = keras.layers.LayerNormalization(epsilon=1e-6, name="complexity_norm")(complexity_x)
        complexity_x = keras.layers.Dense(96, activation="relu", name="complexity_dense_1")(complexity_x)
        complexity_x = keras.layers.Dropout(dropout)(complexity_x)
        complexity_x = keras.layers.Dense(48, activation="relu", name="complexity_dense_2")(complexity_x)
        x = keras.layers.Concatenate(name="shared_plus_complexity")([x, complexity_x])
        x = keras.layers.Dense(160, activation="relu", name="complexity_fusion_dense")(x)
        x = keras.layers.Dropout(dropout)(x)

    if use_start_stop_contrast and start_stop_parameter_indices:
        ss_x = StartStopContrastFeatures(
            start_stop_parameter_indices,
            half_width=25,
            name="start_stop_contrast_features",
        )(profile_x)
        ss_x = keras.layers.LayerNormalization(epsilon=1e-6, name="start_stop_contrast_norm")(ss_x)
        ss_x = keras.layers.Dense(64, activation="relu", name="start_stop_contrast_dense_1")(ss_x)
        ss_x = keras.layers.Dropout(dropout)(ss_x)
        ss_x = keras.layers.Dense(32, activation="relu", name="start_stop_contrast_dense_2")(ss_x)
        x = keras.layers.Concatenate(name="shared_plus_start_stop_contrast")([x, ss_x])
        x = keras.layers.Dense(176, activation="relu", name="start_stop_contrast_fusion_dense")(x)
        x = keras.layers.Dropout(dropout)(x)

    inputs: Any = profile_inputs
    if use_sequence_branch or use_motif_detection_branch:
        sequence_inputs = keras.layers.Input(shape=(sequence_length, 4), name="sequence")
        sequence_parts = []
        if use_sequence_branch:
            seq_x = keras.layers.Conv1D(64, 9, padding="same", activation="relu", name="motif_conv_9")(sequence_inputs)
            seq_x = keras.layers.MaxPooling1D(2, name="motif_pool_1")(seq_x)
            seq_x = keras.layers.Conv1D(96, 15, padding="same", activation="relu", name="motif_conv_15")(seq_x)
            seq_x = keras.layers.MaxPooling1D(2, name="motif_pool_2")(seq_x)
            seq_x = keras.layers.Conv1D(128, 21, padding="same", activation="relu", name="motif_conv_21")(seq_x)
            seq_x = keras.layers.GlobalMaxPooling1D(name="motif_global_max")(seq_x)
            seq_x = keras.layers.Dense(96, activation="relu", name="motif_dense")(seq_x)
            seq_x = keras.layers.Dropout(dropout)(seq_x)
            sequence_parts.append(seq_x)
        if use_motif_detection_branch:
            motif_dim = max(8, int(motif_detection_dim))
            motif_hidden = max(16, motif_dim * 2)
            explicit_x = ExplicitMotifFeatures(
                sequence_length=sequence_length,
                name="explicit_motif_features",
            )(sequence_inputs)
            explicit_x = keras.layers.LayerNormalization(epsilon=1e-6, name="explicit_motif_norm")(explicit_x)
            explicit_x = keras.layers.Dense(motif_hidden, activation="relu", name="explicit_motif_dense_1")(explicit_x)
            explicit_x = keras.layers.Dropout(dropout)(explicit_x)
            explicit_x = keras.layers.Dense(motif_dim, activation="relu", name="explicit_motif_dense_2")(explicit_x)
            explicit_x = keras.layers.Dropout(dropout, name="explicit_motif_output_dropout")(explicit_x)
            sequence_parts.append(explicit_x)
        if len(sequence_parts) == 1:
            sequence_x = sequence_parts[0]
        else:
            sequence_x = keras.layers.Concatenate(name="sequence_motif_feature_concat")(sequence_parts)
        x = keras.layers.Concatenate(name="profile_plus_motif")([x, sequence_x])
        x = keras.layers.Dense(192, activation="relu", name="profile_motif_fusion_dense")(x)
        x = keras.layers.Dropout(dropout)(x)
        inputs = {"profile": profile_inputs, "sequence": sequence_inputs}

    if use_kg_branch:
        kg_inputs = keras.layers.Input(shape=(kg_feature_dim,), name="kg_features")
        kg_x = keras.layers.LayerNormalization(epsilon=1e-6, name="kg_feature_norm")(kg_inputs)
        kg_x = keras.layers.Dense(96, activation="relu", name="kg_embedding_dense_1")(kg_x)
        kg_x = keras.layers.Dropout(dropout)(kg_x)
        kg_x = keras.layers.Dense(48, activation="relu", name="kg_embedding_dense_2")(kg_x)
        x = keras.layers.Concatenate(name="profile_plus_kg")([x, kg_x])
        x = keras.layers.Dense(192, activation="relu", name="kg_fusion_dense")(x)
        x = keras.layers.Dropout(dropout)(x)
        if isinstance(inputs, dict):
            inputs["kg_features"] = kg_inputs
        else:
            inputs = {"profile": profile_inputs, "kg_features": kg_inputs}

    kingdom_base = x
    if kingdom_gradient_reversal_lambda > 0:
        kingdom_base = GradientReversal(
            kingdom_gradient_reversal_lambda,
            name="kingdom_gradient_reversal",
        )(kingdom_base)
    kingdom_repr = keras.layers.Dense(96, activation="relu", name="kingdom_tower_dense")(kingdom_base)
    kingdom_repr = keras.layers.Dropout(dropout)(kingdom_repr)
    kingdom_out = keras.layers.Dense(
        max(head_classes["kingdom"], 1),
        activation="softmax",
        dtype="float32",
        name="kingdom",
    )(kingdom_repr)
    outputs: dict = {"kingdom": kingdom_out}

    # Ground-truth kingdom one-hot injected as a conditioning input to the element
    # tower.  At inference time the kingdom is always known from organism metadata,
    # so this is not label leakage — it is an explicit biological prior.
    # The kingdom prediction head above still runs from the shared representation x
    # (no shortcut), so it keeps its monitoring value.
    if kingdom_label_conditioning:
        n_kingdom_classes = max(head_classes["kingdom"], 1)
        kingdom_label_input = keras.layers.Input(
            shape=(n_kingdom_classes,), name="kingdom_label"
        )
        if not isinstance(inputs, dict):
            inputs = {"profile": inputs}
        inputs["kingdom_label"] = kingdom_label_input

    organism_base = x
    if organism_gradient_reversal_lambda > 0:
        organism_base = GradientReversal(
            organism_gradient_reversal_lambda,
            name="organism_gradient_reversal",
        )(organism_base)

    # The May-19 best animalia run conditioned downstream towers on kingdom_repr
    # even when the scoped kingdom head had one class. Keep that behavior
    # available for exact reproduction. With gradient reversal, keep the
    # adversarial organism classifier directly on the shared representation so
    # it cannot bypass invariance through kingdom_repr.
    if organism_gradient_reversal_lambda <= 0 and (legacy_kingdom_conditioning or head_classes["kingdom"] > 1):
        organism_conditioning = [x, kingdom_repr]
    else:
        organism_conditioning = [organism_base]
    organism_repr = keras.layers.Concatenate(name="organism_conditioned_features")(organism_conditioning) if len(organism_conditioning) > 1 else organism_conditioning[0]
    organism_repr = keras.layers.Dense(128, activation="relu", name="organism_tower_dense")(organism_repr)
    organism_repr = keras.layers.Dropout(dropout)(organism_repr)

    if kingdom_specific_organism_head and kingdom_organism_head_classes:
        # One softmax head per kingdom; each is supervised only on its kingdom's
        # samples via sample_weight masking in the data generator.
        for k_name, n_org_k in kingdom_organism_head_classes.items():
            outputs[f"organism_{k_name}"] = keras.layers.Dense(
                n_org_k,
                activation="softmax",
                dtype="float32",
                name=f"organism_{k_name}",
            )(organism_repr)
    else:
        organism_out = keras.layers.Dense(
            head_classes["organism"],
            activation="softmax",
            dtype="float32",
            name="organism",
        )(organism_repr)
        outputs["organism"] = organism_out

    element_conditioning = [x]
    if kingdom_gradient_reversal_lambda <= 0 and (legacy_kingdom_conditioning or head_classes["kingdom"] > 1):
        element_conditioning.append(kingdom_repr)
    if kingdom_label_conditioning:
        element_conditioning.append(kingdom_label_input)
    if condition_element_on_organism:
        # organism_repr (128-dim tower) is available regardless of head mode
        element_conditioning.append(organism_repr)
    element_repr = keras.layers.Concatenate(name="element_conditioned_features")(element_conditioning)
    element_repr = keras.layers.Dense(192, activation="relu", name="element_tower_dense_1")(element_repr)
    element_repr = keras.layers.Dropout(dropout)(element_repr)
    element_repr = keras.layers.Dense(128, activation="relu", name="element_tower_dense_2")(element_repr)
    element_repr = keras.layers.Dropout(dropout)(element_repr)

    if kingdom_specific_element_head and kingdom_element_head_classes:
        for k_name, n_elem_k in kingdom_element_head_classes.items():
            outputs[f"element_{k_name}"] = keras.layers.Dense(
                n_elem_k, activation="softmax", dtype="float32", name=f"element_{k_name}",
            )(element_repr)
    else:
        element_out = keras.layers.Dense(
            head_classes["element"],
            activation="softmax",
            dtype="float32",
            name="element",
        )(element_repr)
        outputs["element"] = element_out

    if HUMAN_SPECIFIC_HEAD in head_classes:
        human_repr = keras.layers.Dense(96, activation="relu", name="human_specific_element_dense_1")(element_repr)
        human_repr = keras.layers.Dropout(dropout)(human_repr)
        human_repr = keras.layers.Dense(64, activation="relu", name="human_specific_element_dense_2")(human_repr)
        human_repr = keras.layers.Dropout(dropout)(human_repr)
        outputs[HUMAN_SPECIFIC_HEAD] = keras.layers.Dense(
            head_classes[HUMAN_SPECIFIC_HEAD],
            activation="softmax",
            dtype="float32",
            name=HUMAN_SPECIFIC_HEAD,
        )(human_repr)
        if UTR_SUBTYPE_HEAD in head_classes:
            utr_sub = keras.layers.Dense(32, activation="relu", name="utr_subtype_dense")(human_repr)
            utr_sub = keras.layers.Dropout(dropout)(utr_sub)
            outputs[UTR_SUBTYPE_HEAD] = keras.layers.Dense(
                head_classes[UTR_SUBTYPE_HEAD],
                activation="softmax",
                dtype="float32",
                name=UTR_SUBTYPE_HEAD,
            )(utr_sub)
    if CDS_DETECTOR_HEAD in head_classes:
        cds_det = keras.layers.Dense(32, activation="relu", name="cds_detector_dense")(element_repr)
        cds_det = keras.layers.Dropout(dropout)(cds_det)
        outputs[CDS_DETECTOR_HEAD] = keras.layers.Dense(
            head_classes[CDS_DETECTOR_HEAD],
            activation="softmax",
            dtype="float32",
            name=CDS_DETECTOR_HEAD,
        )(cds_det)
    if "aux_organism_element" in head_classes:
        aux_repr = keras.layers.Dense(160, activation="relu", name="aux_organism_element_dense")(element_repr)
        aux_repr = keras.layers.Dropout(dropout)(aux_repr)
        outputs["aux_organism_element"] = keras.layers.Dense(
            head_classes["aux_organism_element"],
            activation="softmax",
            dtype="float32",
            name="aux_organism_element",
        )(aux_repr)
    if "aux_element_coarse" in head_classes:
        coarse_repr = keras.layers.Dense(96, activation="relu", name="aux_element_coarse_dense")(element_repr)
        coarse_repr = keras.layers.Dropout(dropout)(coarse_repr)
        outputs["aux_element_coarse"] = keras.layers.Dense(
            head_classes["aux_element_coarse"],
            activation="softmax",
            dtype="float32",
            name="aux_element_coarse",
        )(coarse_repr)
    if "bio_element_group" in head_classes:
        bio_repr = keras.layers.Dense(96, activation="relu", name="bio_element_group_dense")(element_repr)
        bio_repr = keras.layers.Dropout(dropout)(bio_repr)
        outputs["bio_element_group"] = keras.layers.Dense(
            head_classes["bio_element_group"],
            activation="softmax",
            dtype="float32",
            name="bio_element_group",
        )(bio_repr)
    if BOUNDARY_CONTEXT_HEAD in head_classes:
        boundary_repr = keras.layers.Dense(96, activation="relu", name="boundary_context_dense")(element_repr)
        boundary_repr = keras.layers.Dropout(dropout)(boundary_repr)
        outputs[BOUNDARY_CONTEXT_HEAD] = keras.layers.Dense(
            head_classes[BOUNDARY_CONTEXT_HEAD],
            activation="softmax",
            dtype="float32",
            name=BOUNDARY_CONTEXT_HEAD,
        )(boundary_repr)
    for pair_head in PAIR_ELEMENT_HEADS:
        if pair_head in head_classes:
            pair_repr = keras.layers.Dense(64, activation="relu", name=f"{pair_head}_dense")(element_repr)
            pair_repr = keras.layers.Dropout(dropout)(pair_repr)
            outputs[pair_head] = keras.layers.Dense(
                head_classes[pair_head],
                activation="softmax",
                dtype="float32",
                name=pair_head,
            )(pair_repr)
    return keras.Model(inputs=inputs, outputs=outputs, name="taxonomy_multitask_profile_model")


def build_framework_schema(encoders: dict[str, Any]) -> dict[str, Any]:
    element_labels = ordered_labels(encoders["element"])
    organism_labels = ordered_labels(encoders["organism"])
    kingdom_labels = ordered_labels(encoders["kingdom"])

    universal_present = [e for e in UNIVERSAL_ELEMENTS if e in element_labels]
    human_specific_present = [e for e in HUMAN_SPECIFIC_ELEMENTS if e in element_labels]

    return {
        "schema_version": 1,
        "data_type": "normalized genomic biophysical profile tensor",
        "input_shape": {
            "axis_order_on_disk": ["sample", "parameter", "position"],
            "axis_order_model": ["sample", "position", "parameter"],
            "parameters": ordered_labels(encoders["parameter"]),
            "n_positions": encoders.get("n_positions"),
            "param_n_positions": encoders.get("param_n_positions", {}),
        },
        "heads": {
            "kingdom": {
                "label_array": "Y_kingdom.npy",
                "labels": kingdom_labels,
                "scope": "current eukaryotic kingdoms",
                "extension_note": "Add prokaryotic/superkingdom labels here when mixing bacteria with eukaryotes.",
            },
            "organism": {
                "label_array": "Y_organism.npy",
                "labels": organism_labels,
                "scope": "current organisms in dataset",
                "extension_note": "Can grow from 11 eukaryotes to include bacterial species if trained jointly.",
            },
            "element": {
                "label_array": "Y_element.npy",
                "labels": element_labels,
                "universal_elements": universal_present,
                "human_specific_elements": human_specific_present,
                "requested_universal_elements": UNIVERSAL_ELEMENTS,
                "requested_human_specific_elements": HUMAN_SPECIFIC_ELEMENTS,
                "scope": "8 universal sites plus human-specific UTR/enhancer/splice-boundary sites when present",
                "coarse_groups": {
                    "order": COARSE_ELEMENT_ORDER,
                    "mapping": ELEMENT_COARSE,
                },
            },
        },
        "planned_future_heads": {
            "prokaryotic_species": {
                "label_array": "Y_prokaryotic_species.npy",
                "labels": "49 bacterial species when available",
                "scope": "bacterial species identity for future TSS work",
            },
            "tss": {
                "label_array": "Y_tss.npy",
                "labels": ["non_tss", "tss"],
                "scope": "future bacterial transcription start site prediction",
            },
        },
        "paper_guided_features": {
            "table_window_branch": {
                "source": "Supplementary Tables S3 and S4 in papers_for_reference/su.pdf",
                "kingdoms": sorted(TABLE_WINDOWS),
                "elements": TABLE_SITE_ORDER,
                "parameters": sorted(next(iter(TABLE_WINDOWS.values())).keys()),
                "description": "Fixed mean-pooled windows for kingdom/site/parameter-specific structural and energetic signal regions.",
            },
            "complexity_feature_branch": {
                "features": [
                    "mean", "std", "skewness", "kurtosis", "min", "max", "range",
                    "shannon_entropy", "spectral_mean", "spectral_std",
                    "cfs_mean", "cfs_std", "cfs_skewness", "cfs_kurtosis",
                ],
                "description": "Per-sample physicochemical profile summary inspired by entropy/moment/CFS analyses.",
            },
        },
    }


def crop_positions(X_raw: np.ndarray, crop_window: int) -> tuple[np.ndarray, int, int]:
    original_n_positions = int(X_raw.shape[2])
    if crop_window <= 0:
        return X_raw, 0, original_n_positions
    center = original_n_positions // 2
    start = max(0, center - crop_window)
    end = min(original_n_positions, center + crop_window + 1)
    return X_raw[:, :, start:end], start, original_n_positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train taxonomy-aware kingdom/organism/element heads")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", choices=["global", "kingdom", "organism"], default="global",
                        help="Train on all rows, one kingdom, or one organism.")
    parser.add_argument("--scope-name", default=None,
                        help="Required for --scope kingdom/organism, e.g. animalia or human.")
    parser.add_argument("--holdout-kingdom", default=None,
                        choices=["animalia", "fungi", "plantae", "protista"],
                        help="Leave-one-kingdom-out: exclude this kingdom from train+val and "
                             "evaluate zero-shot element transfer on its test split.")
    parser.add_argument("--element-set", choices=["universal", "all"], default="universal",
                        help="Use only the 8 universal elements by default; 'all' also includes human-specific sites.")
    parser.add_argument("--human-specific-element-head", dest="human_specific_element_head",
                        action="store_true",
                        help="Keep universal element prediction separate and add a masked head for human-only extra sites.")
    parser.add_argument("--no-human-specific-element-head", dest="human_specific_element_head",
                        action="store_false")
    parser.set_defaults(human_specific_element_head=False)
    parser.add_argument("--utr-subtype-head", dest="utr_subtype_head", action="store_true",
                        help="Add a 3utr vs 5utr sub-head branching from the human-specific head (requires --human-specific-element-head).")
    parser.add_argument("--no-utr-subtype-head", dest="utr_subtype_head", action="store_false")
    parser.set_defaults(utr_subtype_head=False)
    parser.add_argument("--cds-detector-head", dest="cds_detector_head", action="store_true",
                        help="Add a binary CDS vs non-CDS head; removes CDS from the main element head.")
    parser.add_argument("--no-cds-detector-head", dest="cds_detector_head", action="store_false")
    parser.set_defaults(cds_detector_head=False)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-stages", type=int, default=3)
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--stem-kernel", type=int, default=7)
    parser.add_argument("--block-kernel", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="AdamW weight decay; 0 uses plain Adam.")
    parser.add_argument("--element-label-smoothing", type=float, default=0.0,
                        help="Label smoothing for the element focal loss.")
    parser.add_argument("--crop-window", type=int, default=0,
                        help="Use 0 for all positions, or e.g. 100 for ±100 bp around the boundary.")
    parser.add_argument("--element-loss-weight", type=float, default=2.0)
    parser.add_argument("--element-focal-loss-gamma", type=float, default=0.0,
                        help="Use focal loss on the main element head when > 0; 0 keeps cross-entropy.")
    parser.add_argument("--human-specific-element-loss-weight", type=float, default=0.8)
    parser.add_argument("--utr-subtype-loss-weight", type=float, default=0.2)
    parser.add_argument("--cds-detector-loss-weight", type=float, default=0.5)
    parser.add_argument("--organism-loss-weight", type=float, default=1.0)
    parser.add_argument("--organism-gradient-reversal-lambda", type=float, default=0.0,
                        help="When >0, train the organism head through a gradient reversal layer to discourage organism-predictive shared features.")
    parser.add_argument("--kingdom-gradient-reversal-lambda", type=float, default=0.0,
                        help="When >0, train the kingdom head through a gradient reversal layer to discourage kingdom-predictive shared features. "
                             "Pairs with --kingdom-label-conditioning so the element tower still receives kingdom identity via ground-truth input.")
    parser.add_argument("--grl-warmup-epochs", type=int, default=0,
                        help="Ramp GRL lambda 0->target over the first N epochs (0=off). "
                             "Stabilizes the no-sequence arms (biophys/profiles_kg) that blow up "
                             "under a fixed lambda; final lambda is unchanged.")
    parser.add_argument("--kingdom-loss-weight", type=float, default=0.5)
    parser.add_argument("--aux-organism-element-loss-weight", type=float, default=0.6)
    parser.add_argument("--aux-organism-element-head", dest="aux_organism_element_head", action="store_true",
                        help="Add an auxiliary head for valid organism::element combinations.")
    parser.add_argument("--no-aux-organism-element-head", dest="aux_organism_element_head", action="store_false")
    parser.set_defaults(aux_organism_element_head=True)
    parser.add_argument("--aux-element-coarse-loss-weight", type=float, default=0.5)
    parser.add_argument("--aux-element-coarse-head", dest="aux_element_coarse_head", action="store_true",
                        help="Add an auxiliary coarse element-family head.")
    parser.add_argument("--no-aux-element-coarse-head", dest="aux_element_coarse_head", action="store_false")
    parser.set_defaults(aux_element_coarse_head=True)
    parser.add_argument("--bio-hierarchy-heads", dest="bio_hierarchy_heads", action="store_true",
                        help="Add biological group and pair-specific element heads.")
    parser.add_argument("--no-bio-hierarchy-heads", dest="bio_hierarchy_heads", action="store_false")
    parser.set_defaults(bio_hierarchy_heads=True)
    parser.add_argument("--bio-element-group-loss-weight", type=float, default=0.8)
    parser.add_argument("--pair-element-loss-weight", type=float, default=0.4)
    parser.add_argument("--boundary-context-head", dest="boundary_context_head", action="store_true",
                        help="Add a masked head for gene_boundary/exon_boundary/prom confusions.")
    parser.add_argument("--no-boundary-context-head", dest="boundary_context_head", action="store_false")
    parser.set_defaults(boundary_context_head=True)
    parser.add_argument("--boundary-context-loss-weight", type=float, default=0.6)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--balance-elements", dest="balance_elements", action="store_true",
                        help="Downsample every split so all element classes have equal counts.")
    parser.add_argument("--no-balance-elements", dest="balance_elements", action="store_false")
    parser.set_defaults(balance_elements=True)
    parser.add_argument("--max-balanced-per-element", type=int, default=0,
                        help="Optional cap after min-count balancing; 0 means no extra cap.")
    parser.add_argument("--max-balanced-per-human-specific-element", type=int, default=0,
                        help="Optional cap per human-specific class when the independent human head is enabled.")
    parser.add_argument("--balance-organism-element", dest="balance_organism_element",
                        action="store_true",
                        help="Balance jointly by organism-element instead of element only.")
    parser.add_argument("--no-balance-organism-element", dest="balance_organism_element",
                        action="store_false")
    parser.set_defaults(balance_organism_element=False)
    parser.add_argument("--max-balanced-per-organism-element", type=int, default=0,
                        help="Per-(organism,element) cap when --balance-organism-element is set; 0 = no cap.")
    parser.add_argument("--max-balanced-per-organism-element-train", type=int, default=0,
                        help="Train split override for --max-balanced-per-organism-element; 0 = use shared cap.")
    parser.add_argument("--max-balanced-per-organism-element-val", type=int, default=0,
                        help="Validation split override for --max-balanced-per-organism-element; 0 = use shared cap.")
    parser.add_argument("--max-balanced-per-organism-element-test", type=int, default=0,
                        help="Test split override for --max-balanced-per-organism-element; 0 = use shared cap.")
    parser.add_argument("--min-balanced-per-organism-element", type=int, default=0,
                        help="Upsample (with replacement) any (organism,element) combo below this floor; 0 = no floor.")
    parser.add_argument("--min-balanced-per-organism-element-train", type=int, default=0,
                        help="Train split override for --min-balanced-per-organism-element; 0 = use shared floor.")
    parser.add_argument("--min-balanced-per-organism-element-val", type=int, default=0,
                        help="Validation split override for --min-balanced-per-organism-element; 0 = use shared floor.")
    parser.add_argument("--min-balanced-per-organism-element-test", type=int, default=0,
                        help="Test split override for --min-balanced-per-organism-element; 0 = use shared floor.")
    parser.add_argument("--balance-kingdom-element", dest="balance_kingdom_element",
                        action="store_true",
                        help="Balance jointly by kingdom-element instead of element only.")
    parser.add_argument("--no-balance-kingdom-element", dest="balance_kingdom_element",
                        action="store_false")
    parser.set_defaults(balance_kingdom_element=False)
    parser.add_argument("--max-balanced-per-kingdom-element", type=int, default=0,
                        help="Per-(kingdom,element) cap when --balance-kingdom-element is set; 0 = no cap.")
    parser.add_argument("--max-balanced-per-kingdom-element-train", type=int, default=0,
                        help="Train split override for --max-balanced-per-kingdom-element; 0 = use shared cap.")
    parser.add_argument("--max-balanced-per-kingdom-element-val", type=int, default=0,
                        help="Validation split override for --max-balanced-per-kingdom-element; 0 = use shared cap.")
    parser.add_argument("--max-balanced-per-kingdom-element-test", type=int, default=0,
                        help="Test split override for --max-balanced-per-kingdom-element; 0 = use shared cap.")
    parser.add_argument("--organism-profile-normalization", dest="organism_profile_normalization",
                        action="store_true",
                        help="Z-score each profile by train-split mean/std for its organism.")
    parser.add_argument("--no-organism-profile-normalization", dest="organism_profile_normalization",
                        action="store_false")
    parser.set_defaults(organism_profile_normalization=False)
    parser.add_argument("--normalize-heldout-from-test", action="store_true",
                        help="LOKO only: estimate held-out organisms' profile normalization from "
                             "their own (test) profiles instead of identity. Transductive input "
                             "standardization — no labels used, prediction stays zero-shot. Tests "
                             "whether biophys transfer failure is the representation vs un-normalized input.")
    parser.add_argument("--organism-normalization-eps", type=float, default=1e-6)
    parser.add_argument("--organism-normalization-chunk-size", type=int, default=2048)
    parser.add_argument("--table-window-branch", dest="table_window_branch", action="store_true",
                        help="Use Supplementary Table S3/S4 windows as a pooled feature branch.")
    parser.add_argument("--no-table-window-branch", dest="table_window_branch", action="store_false")
    parser.set_defaults(table_window_branch=True)
    parser.add_argument("--table-center", type=int, default=250,
                        help="Coordinate of the centered site in the paper tables.")
    parser.add_argument("--complexity-feature-branch", dest="complexity_feature_branch", action="store_true",
                        help="Use entropy/moment/cumulative-Fourier summary features.")
    parser.add_argument("--no-complexity-feature-branch", dest="complexity_feature_branch", action="store_false")
    parser.set_defaults(complexity_feature_branch=True)
    parser.add_argument("--profile-capsule-branch", dest="profile_capsule_branch", action="store_true",
                        help="Add compact capsule-style pooling over profile feature maps.")
    parser.add_argument("--no-profile-capsule-branch", dest="profile_capsule_branch", action="store_false")
    parser.set_defaults(profile_capsule_branch=False)
    parser.add_argument("--profile-capsule-num-capsules", type=int, default=16)
    parser.add_argument("--profile-capsule-dim", type=int, default=8)
    parser.add_argument("--sequence-branch", dest="sequence_branch", action="store_true",
                        help="Add a 501 bp one-hot CNN motif branch fused with profile features.")
    parser.add_argument("--no-sequence-branch", dest="sequence_branch", action="store_false")
    parser.set_defaults(sequence_branch=False)
    parser.add_argument("--motif-detection-branch", dest="motif_detection_branch", action="store_true",
                        help="Add explicit position-aware motif and composition features from the sequence.")
    parser.add_argument("--no-motif-detection-branch", dest="motif_detection_branch", action="store_false")
    parser.set_defaults(motif_detection_branch=False)
    parser.add_argument("--motif-detection-dim", type=int, default=32,
                        help="Output dimension for explicit motif features before sequence/profile fusion.")
    parser.add_argument("--sequence-manifest", type=Path, default=Path("sequence_manifest.parquet"))
    parser.add_argument("--sequence-root", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=501)
    parser.add_argument("--preload-sequences", dest="preload_sequences", action="store_true",
                        help="Load all sequences into RAM at startup to eliminate per-batch file I/O.")
    parser.add_argument("--preload-profiles", dest="preload_profiles", action="store_true",
                        help="Load X_raw profile array into RAM at startup to eliminate per-batch NFS I/O.")
    parser.add_argument("--kg-branch", dest="kg_branch", action="store_true",
                        help="Fuse knowledge-graph node feature embeddings into the model.")
    parser.add_argument("--no-kg-branch", dest="kg_branch", action="store_false")
    parser.set_defaults(kg_branch=False)
    parser.add_argument("--kg-dir", type=Path, default=Path("knowledge_graph"))
    parser.add_argument("--legacy-kingdom-conditioning", dest="legacy_kingdom_conditioning",
                        action="store_true",
                        help="Reproduce the May-19 best architecture by conditioning organism/element towers on kingdom_repr even for one-class kingdom scopes.")
    parser.add_argument("--no-legacy-kingdom-conditioning", dest="legacy_kingdom_conditioning",
                        action="store_false")
    parser.set_defaults(legacy_kingdom_conditioning=False)
    parser.add_argument("--kingdom-label-conditioning", dest="kingdom_label_conditioning",
                        action="store_true",
                        help="Inject the ground-truth kingdom one-hot as an explicit input to the element tower. "
                             "Kingdom is always known at inference time from organism metadata, so this is a "
                             "biologically principled prior, not label leakage.")
    parser.add_argument("--no-kingdom-label-conditioning", dest="kingdom_label_conditioning",
                        action="store_false")
    parser.set_defaults(kingdom_label_conditioning=False)
    parser.add_argument("--condition-element-on-organism", dest="condition_element_on_organism",
                        action="store_true",
                        help="Feed organism tower representation into the element tower.")
    parser.add_argument("--no-condition-element-on-organism", dest="condition_element_on_organism",
                        action="store_false",
                        help="Predict organism as an auxiliary task without directly conditioning element prediction on organism representation.")
    parser.set_defaults(condition_element_on_organism=True)
    parser.add_argument("--collapse-ambiguous-boundaries", action="store_true",
                        help="Collapse es/ee into exon_boundary and gs/ge into gene_boundary.")
    parser.add_argument("--paper-parameter-scaling", dest="paper_parameter_scaling", action="store_true",
                        help="Weight start/stop-discriminative parameters highlighted in the paper.")
    parser.add_argument("--no-paper-parameter-scaling", dest="paper_parameter_scaling", action="store_false")
    parser.set_defaults(paper_parameter_scaling=True)
    parser.add_argument("--start-stop-contrast-branch", dest="start_stop_contrast_branch", action="store_true",
                        help="Add focused H-bond/stacking/solvation/BP/intra/backbone contrast features.")
    parser.add_argument("--no-start-stop-contrast-branch", dest="start_stop_contrast_branch", action="store_false")
    parser.set_defaults(start_stop_contrast_branch=True)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load weights and run the test eval only "
                             "(produces per_kingdom_element_accuracy.json etc.).")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Checkpoint to load for --eval-only (default: <out>/checkpoints/best_model.keras).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in --out/checkpoints/")
    parser.add_argument("--kingdom-specific-organism-head", dest="kingdom_specific_organism_head",
                        action="store_true",
                        help="Train a separate organism softmax head per kingdom instead of one global head.")
    parser.set_defaults(kingdom_specific_organism_head=False)
    parser.add_argument("--kingdom-specific-element-head", dest="kingdom_specific_element_head",
                        action="store_true",
                        help="Train a separate element logit head per kingdom, hard-routed by ground-truth kingdom "
                             "one-hot. Requires --kingdom-label-conditioning.")
    parser.set_defaults(kingdom_specific_element_head=False)
    args = parser.parse_args()

    if tf is None:
        raise ModuleNotFoundError("TensorFlow is required. Activate the genome_reader_tf environment.")

    tf.keras.utils.set_random_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "X_raw.npy",
        "Y_element.npy",
        "Y_organism.npy",
        "Y_kingdom.npy",
        "train_idx.npy",
        "val_idx.npy",
        "test_idx.npy",
        "encoders.json",
    ]
    missing = [name for name in required if not (args.dataset / name).exists()]
    if missing:
        raise FileNotFoundError(f"Dataset is missing required files: {missing}")

    X_raw = np.load(args.dataset / "X_raw.npy", mmap_mode="r")
    if args.preload_profiles:
        n, mem_gb = X_raw.shape[0], X_raw.nbytes / 1e9
        print(f"Pre-loading {n:,} profiles into RAM (~{mem_gb:.1f} GB float32)...")
        X_raw = np.array(X_raw, dtype=np.float32)
        print("Profile pre-load complete.")
    X_raw, profile_offset, original_n_positions = crop_positions(X_raw, args.crop_window)
    labels = {
        "element": np.load(args.dataset / "Y_element.npy"),
        "organism": np.load(args.dataset / "Y_organism.npy"),
        "kingdom": np.load(args.dataset / "Y_kingdom.npy"),
    }
    train_idx = np.load(args.dataset / "train_idx.npy")
    val_idx = np.load(args.dataset / "val_idx.npy")
    test_idx = np.load(args.dataset / "test_idx.npy")
    with open(args.dataset / "encoders.json") as fh:
        encoders = json.load(fh)

    schema = build_framework_schema(encoders)
    save_json(args.out / "framework_schema.json", schema)

    if getattr(args, "holdout_kingdom", None):
        # Leave-one-kingdom-out: train+validate only on the OTHER kingdoms; evaluate
        # zero-shot element transfer on the held-out kingdom's test split.
        kid = int(encoders["kingdom"][args.holdout_kingdom])
        kk = labels["kingdom"]
        train_idx = train_idx[kk[train_idx] != kid]
        val_idx = val_idx[kk[val_idx] != kid]
        test_idx = test_idx[kk[test_idx] == kid]
        print(f"LEAVE-ONE-KINGDOM-OUT: holding out '{args.holdout_kingdom}' (id {kid}); "
              f"train={len(train_idx):,} val={len(val_idx):,} test[held-out]={len(test_idx):,}",
              flush=True)

    train_idx, scope_train = filter_indices_by_scope(
        train_idx, labels, encoders, args.scope, args.scope_name)
    val_idx, scope_val = filter_indices_by_scope(
        val_idx, labels, encoders, args.scope, args.scope_name)
    test_idx, scope_test = filter_indices_by_scope(
        test_idx, labels, encoders, args.scope, args.scope_name)

    train_idx, element_set_train = filter_indices_by_element_set(
        train_idx, labels, encoders, args.element_set,
        include_human_specific_for_head=bool(args.human_specific_element_head))
    val_idx, element_set_val = filter_indices_by_element_set(
        val_idx, labels, encoders, args.element_set,
        include_human_specific_for_head=bool(args.human_specific_element_head))
    test_idx, element_set_test = filter_indices_by_element_set(
        test_idx, labels, encoders, args.element_set,
        include_human_specific_for_head=bool(args.human_specific_element_head))
    scope_info = {
        "scope": args.scope,
        "scope_name": args.scope_name,
        "train": scope_train,
        "val": scope_val,
        "test": scope_test,
        "element_set": {
            "train": element_set_train,
            "val": element_set_val,
            "test": element_set_test,
        },
    }

    labels, active_label_names, label_remap_info = remap_labels_to_active_scope(
        labels,
        encoders,
        train_idx,
        val_idx,
        test_idx,
        heads=["element", "organism", "kingdom"],
    )
    scope_info["label_remap"] = label_remap_info

    element_names = active_label_names["element"]
    element_collapse_info: dict[str, Any] = {"enabled": False}
    if args.collapse_ambiguous_boundaries:
        collapsed_element_labels, element_names, element_collapse_info = collapse_ambiguous_element_labels(
            labels["element"],
            element_names,
        )
        labels["element"] = collapsed_element_labels
        active_label_names["element"] = element_names
    scope_info["element_label_collapse"] = element_collapse_info

    sample_weight_masks: dict[str, np.ndarray] = {}
    human_specific_schema: dict[str, Any] | None = None
    main_element_mask: np.ndarray | None = None
    utr_subtype_schema: dict[str, Any] | None = None
    cds_detector_schema: dict[str, Any] | None = None
    cds_det_labels: np.ndarray | None = None
    cds_det_mask: np.ndarray | None = None
    # Build CDS detector BEFORE any splits so original element IDs are intact.
    if args.cds_detector_head:
        cds_det_labels, cds_det_mask, cds_detector_schema = build_cds_detector_labels(
            labels["element"], element_names,
        )
        if not cds_detector_schema.get("enabled"):
            print(f"CDS detector head disabled: {cds_detector_schema.get('reason', 'unknown')}")
            cds_detector_schema = None
    if args.human_specific_element_head:
        # Build utr_subtype BEFORE split removes UTR elements from labels["element"].
        utr_sub_labels: np.ndarray | None = None
        utr_sub_mask: np.ndarray | None = None
        if args.utr_subtype_head:
            utr_sub_labels, utr_sub_mask, utr_subtype_schema = build_utr_subtype_labels(
                labels["element"],
                labels["organism"],
                element_names,
                active_label_names["organism"],
            )
            if not utr_subtype_schema.get("enabled"):
                print(f"UTR subtype head disabled: {utr_subtype_schema.get('reason', 'unknown')}")
                utr_subtype_schema = None

        (
            main_element_labels,
            human_specific_labels,
            main_element_mask,
            human_specific_mask,
            main_element_names,
            human_specific_schema,
        ) = split_human_specific_element_head(
            labels["element"],
            labels["organism"],
            element_names,
            active_label_names["organism"],
        )
        if human_specific_schema.get("enabled"):
            labels["element"] = main_element_labels
            labels[HUMAN_SPECIFIC_HEAD] = human_specific_labels
            sample_weight_masks["element"] = main_element_mask
            sample_weight_masks[HUMAN_SPECIFIC_HEAD] = human_specific_mask
            element_names = main_element_names
            active_label_names["element"] = element_names
            active_label_names[HUMAN_SPECIFIC_HEAD] = [
                item["label"] for item in human_specific_schema["labels"]
            ]
            save_json(args.out / "human_specific_element_label_schema.json", human_specific_schema)
            if utr_subtype_schema is not None and utr_sub_labels is not None:
                labels[UTR_SUBTYPE_HEAD] = utr_sub_labels
                sample_weight_masks[UTR_SUBTYPE_HEAD] = utr_sub_mask
                active_label_names[UTR_SUBTYPE_HEAD] = [
                    item["label"] for item in utr_subtype_schema["labels"]
                ]
                save_json(args.out / "utr_subtype_label_schema.json", utr_subtype_schema)
        else:
            print(f"Human-specific element head disabled: {human_specific_schema.get('reason', 'no active samples')}")
            human_specific_schema = None
            utr_subtype_schema = None
            main_element_mask = None

    # Remove CDS from the main element head; downstream heads use the updated mask.
    cds_only_mask: np.ndarray | None = None
    if args.cds_detector_head and cds_detector_schema is not None:
        new_element_labels, cds_removed_mask, new_element_names = split_cds_from_element_head(
            labels["element"], element_names, main_element_mask,
        )
        labels["element"] = new_element_labels
        element_names = new_element_names
        active_label_names["element"] = new_element_names
        main_element_mask = cds_removed_mask
        sample_weight_masks["element"] = cds_removed_mask
        labels[CDS_DETECTOR_HEAD] = cds_det_labels
        sample_weight_masks[CDS_DETECTOR_HEAD] = cds_det_mask
        active_label_names[CDS_DETECTOR_HEAD] = ["non_cds", "cds"]
        # Used only to append CDS samples back after organism-element balancing.
        cds_only_mask = (np.asarray(cds_det_labels) == 1).astype(np.float32)
        save_json(args.out / "cds_detector_label_schema.json", cds_detector_schema)

    combo_schema: dict[str, Any] | None = None
    if args.aux_organism_element_head:
        organism_names = active_label_names["organism"]
        combo_labels, combo_schema = build_combo_labels(
            labels["organism"],
            labels["element"],
            organism_names,
            element_names,
        )
        labels["aux_organism_element"] = combo_labels
        if main_element_mask is not None:
            sample_weight_masks["aux_organism_element"] = main_element_mask
        save_json(args.out / "organism_element_label_schema.json", combo_schema)

    coarse_schema: dict[str, Any] | None = None
    if args.aux_element_coarse_head:
        coarse_labels, coarse_schema = build_coarse_element_labels(labels["element"], element_names)
        labels["aux_element_coarse"] = coarse_labels
        if main_element_mask is not None:
            sample_weight_masks["aux_element_coarse"] = main_element_mask
        save_json(args.out / "coarse_element_label_schema.json", coarse_schema)

    bio_group_schema: dict[str, Any] | None = None
    pair_schemas: dict[str, Any] = {}
    boundary_context_schema: dict[str, Any] | None = None
    if args.bio_hierarchy_heads:
        bio_group_labels, bio_group_schema = build_biological_group_labels(
            labels["element"],
            element_names,
        )
        labels["bio_element_group"] = bio_group_labels
        if main_element_mask is not None:
            sample_weight_masks["bio_element_group"] = main_element_mask
        save_json(args.out / "bio_element_group_label_schema.json", bio_group_schema)

        for pair_head, members in PAIR_ELEMENT_HEADS.items():
            if all(member in element_names for member in members):
                pair_labels, pair_mask, pair_schema = build_pair_element_labels(
                    labels["element"],
                    element_names,
                    pair_head,
                    members,
                )
                labels[pair_head] = pair_labels
                if main_element_mask is not None:
                    pair_mask = pair_mask * main_element_mask
                sample_weight_masks[pair_head] = pair_mask
                pair_schemas[pair_head] = pair_schema
        if pair_schemas:
            save_json(args.out / "pair_element_label_schema.json", {"heads": pair_schemas})

    if args.boundary_context_head:
        boundary_labels, boundary_mask, boundary_context_schema = build_subset_element_labels(
            labels["element"],
            element_names,
            BOUNDARY_CONTEXT_HEAD,
            BOUNDARY_CONTEXT_MEMBERS,
        )
        if main_element_mask is not None:
            boundary_mask = boundary_mask * main_element_mask
        if int(boundary_context_schema["n_classes"]) >= 2 and np.any(boundary_mask > 0):
            labels[BOUNDARY_CONTEXT_HEAD] = boundary_labels
            sample_weight_masks[BOUNDARY_CONTEXT_HEAD] = boundary_mask
            save_json(args.out / "boundary_context_label_schema.json", boundary_context_schema)
        else:
            print(f"Boundary-context head disabled: {boundary_context_schema.get('reason', 'no active samples')}")
            boundary_context_schema = None

    head_classes = {
        "element": len(active_label_names["element"]),
        "organism": len(active_label_names["organism"]),
        "kingdom": len(active_label_names["kingdom"]),
    }
    if human_specific_schema is not None:
        head_classes[HUMAN_SPECIFIC_HEAD] = int(human_specific_schema["n_classes"])
    if utr_subtype_schema is not None:
        head_classes[UTR_SUBTYPE_HEAD] = int(utr_subtype_schema["n_classes"])
    if cds_detector_schema is not None:
        head_classes[CDS_DETECTOR_HEAD] = int(cds_detector_schema["n_classes"])
    if args.aux_organism_element_head and combo_schema is not None:
        head_classes["aux_organism_element"] = int(combo_schema["n_classes"])
    if args.aux_element_coarse_head and coarse_schema is not None:
        head_classes["aux_element_coarse"] = int(coarse_schema["n_classes"])
    if args.bio_hierarchy_heads and bio_group_schema is not None:
        head_classes["bio_element_group"] = int(bio_group_schema["n_classes"])
        for pair_head, pair_schema in pair_schemas.items():
            head_classes[pair_head] = int(pair_schema["n_classes"])
    if boundary_context_schema is not None:
        head_classes[BOUNDARY_CONTEXT_HEAD] = int(boundary_context_schema["n_classes"])
    # Kingdom-specific organism head setup: build per-kingdom organism index maps
    # and remap global organism IDs to within-kingdom local IDs.
    kingdom_organism_head_classes: dict[str, int] = {}
    kingdom_to_global_org_ids: dict[str, list[int]] = {}
    if args.kingdom_specific_organism_head:
        for k_id, k_name in enumerate(active_label_names["kingdom"]):
            k_mask = labels["kingdom"] == k_id
            global_org_ids = sorted(int(x) for x in np.unique(labels["organism"][k_mask]))
            kingdom_to_global_org_ids[k_name] = global_org_ids
            n_org_k = len(global_org_ids)
            kingdom_organism_head_classes[k_name] = n_org_k
            # Remap global organism IDs to local (0-based within kingdom)
            g2l = {g: l for l, g in enumerate(global_org_ids)}
            local_labels = np.zeros(len(labels["organism"]), dtype=np.int32)
            for i in np.where(k_mask)[0]:
                local_labels[i] = g2l[int(labels["organism"][i])]
            labels[f"organism_{k_name}"] = local_labels
            sample_weight_masks[f"organism_{k_name}"] = k_mask.astype(np.float32)
            head_classes[f"organism_{k_name}"] = n_org_k
            org_names_k = [active_label_names["organism"][g] for g in global_org_ids]
            print(f"  Kingdom '{k_name}': {n_org_k} organisms -> {org_names_k}")
        print(f"Kingdom-specific organism heads: {kingdom_organism_head_classes}")

    # Kingdom-specific element head setup: build per-kingdom element index maps
    # and remap global element IDs to within-kingdom local IDs.
    kingdom_element_head_classes: dict[str, int] = {}
    kingdom_to_global_elem_ids: dict[str, list[int]] = {}
    if args.kingdom_specific_element_head:
        for k_id, k_name in enumerate(active_label_names["kingdom"]):
            k_mask = labels["kingdom"] == k_id
            global_elem_ids = sorted(int(x) for x in np.unique(labels["element"][k_mask]))
            kingdom_to_global_elem_ids[k_name] = global_elem_ids
            n_elem_k = len(global_elem_ids)
            kingdom_element_head_classes[k_name] = n_elem_k
            g2l = {g: l for l, g in enumerate(global_elem_ids)}
            local_labels = np.zeros(len(labels["element"]), dtype=np.int32)
            for i in np.where(k_mask)[0]:
                local_labels[i] = g2l[int(labels["element"][i])]
            labels[f"element_{k_name}"] = local_labels
            sample_weight_masks[f"element_{k_name}"] = k_mask.astype(np.float32)
            head_classes[f"element_{k_name}"] = n_elem_k
            elem_names_k = [active_label_names["element"][g] for g in global_elem_ids]
            print(f"  Kingdom '{k_name}': {n_elem_k} elements -> {elem_names_k}")
        print(f"Kingdom-specific element heads: {kingdom_element_head_classes}")

    element_balance: dict[str, Any] = {
        "enabled": bool(args.balance_elements or args.balance_organism_element or args.balance_kingdom_element)
    }
    balance_source_train_idx = train_idx
    balance_source_val_idx = val_idx
    balance_source_test_idx = test_idx
    if args.balance_kingdom_element:
        shared_max_per_combo = (
            args.max_balanced_per_kingdom_element
            if args.max_balanced_per_kingdom_element > 0
            else None
        )
        train_max_per_combo = (
            args.max_balanced_per_kingdom_element_train
            if args.max_balanced_per_kingdom_element_train > 0
            else shared_max_per_combo
        )
        val_max_per_combo = (
            args.max_balanced_per_kingdom_element_val
            if args.max_balanced_per_kingdom_element_val > 0
            else shared_max_per_combo
        )
        test_max_per_combo = (
            args.max_balanced_per_kingdom_element_test
            if args.max_balanced_per_kingdom_element_test > 0
            else shared_max_per_combo
        )
        train_idx, element_balance["train"] = balance_indices_by_kingdom_element(
            train_idx,
            labels["element"],
            labels["kingdom"],
            head_classes["element"],
            head_classes["kingdom"],
            seed=args.seed,
            max_per_combo=train_max_per_combo,
            eligible_mask=main_element_mask,
        )
        val_idx, element_balance["val"] = balance_indices_by_kingdom_element(
            val_idx,
            labels["element"],
            labels["kingdom"],
            head_classes["element"],
            head_classes["kingdom"],
            seed=args.seed + 1,
            max_per_combo=val_max_per_combo,
            eligible_mask=main_element_mask,
        )
        test_idx, element_balance["test"] = balance_indices_by_kingdom_element(
            test_idx,
            labels["element"],
            labels["kingdom"],
            head_classes["element"],
            head_classes["kingdom"],
            seed=args.seed + 2,
            max_per_combo=test_max_per_combo,
            eligible_mask=main_element_mask,
        )
        element_balance["mode"] = "kingdom_element"
    elif args.balance_organism_element:
        shared_max_per_combo = (
            args.max_balanced_per_organism_element
            if args.max_balanced_per_organism_element > 0
            else None
        )
        train_max_per_combo = (
            args.max_balanced_per_organism_element_train
            if args.max_balanced_per_organism_element_train > 0
            else shared_max_per_combo
        )
        val_max_per_combo = (
            args.max_balanced_per_organism_element_val
            if args.max_balanced_per_organism_element_val > 0
            else shared_max_per_combo
        )
        test_max_per_combo = (
            args.max_balanced_per_organism_element_test
            if args.max_balanced_per_organism_element_test > 0
            else shared_max_per_combo
        )
        shared_min_per_combo = (
            args.min_balanced_per_organism_element
            if args.min_balanced_per_organism_element > 0
            else None
        )
        train_min_per_combo = (
            args.min_balanced_per_organism_element_train
            if args.min_balanced_per_organism_element_train > 0
            else shared_min_per_combo
        )
        val_min_per_combo = (
            args.min_balanced_per_organism_element_val
            if args.min_balanced_per_organism_element_val > 0
            else shared_min_per_combo
        )
        test_min_per_combo = (
            args.min_balanced_per_organism_element_test
            if args.min_balanced_per_organism_element_test > 0
            else shared_min_per_combo
        )
        train_idx, element_balance["train"] = balance_indices_by_organism_element(
            train_idx,
            labels["element"],
            labels["organism"],
            head_classes["element"],
            head_classes["organism"],
            seed=args.seed,
            max_per_combo=train_max_per_combo,
            min_per_combo=train_min_per_combo,
            eligible_mask=main_element_mask,
        )
        val_idx, element_balance["val"] = balance_indices_by_organism_element(
            val_idx,
            labels["element"],
            labels["organism"],
            head_classes["element"],
            head_classes["organism"],
            seed=args.seed + 1,
            max_per_combo=val_max_per_combo,
            min_per_combo=val_min_per_combo,
            eligible_mask=main_element_mask,
        )
        test_idx, element_balance["test"] = balance_indices_by_organism_element(
            test_idx,
            labels["element"],
            labels["organism"],
            head_classes["element"],
            head_classes["organism"],
            seed=args.seed + 2,
            max_per_combo=test_max_per_combo,
            min_per_combo=test_min_per_combo,
            eligible_mask=main_element_mask,
        )
        element_balance["mode"] = "organism_element"
    elif args.balance_elements:
        max_per_element = args.max_balanced_per_element if args.max_balanced_per_element > 0 else None
        train_idx, element_balance["train"] = balance_indices_by_label(
            train_idx,
            labels["element"],
            head_classes["element"],
            seed=args.seed,
            max_per_class=max_per_element,
            eligible_mask=main_element_mask,
        )
        val_idx, element_balance["val"] = balance_indices_by_label(
            val_idx,
            labels["element"],
            head_classes["element"],
            seed=args.seed + 1,
            max_per_class=max_per_element,
            eligible_mask=main_element_mask,
        )
        test_idx, element_balance["test"] = balance_indices_by_label(
            test_idx,
            labels["element"],
            head_classes["element"],
            seed=args.seed + 2,
            max_per_class=max_per_element,
            eligible_mask=main_element_mask,
        )
        element_balance["mode"] = "element_only"
    if human_specific_schema is not None and (
        args.balance_elements or args.balance_organism_element or args.balance_kingdom_element
    ):
        max_per_human = (
            args.max_balanced_per_human_specific_element
            if args.max_balanced_per_human_specific_element > 0
            else None
        )
        train_idx, human_train_balance = append_balanced_masked_head_indices(
            train_idx,
            balance_source_train_idx,
            labels[HUMAN_SPECIFIC_HEAD],
            sample_weight_masks[HUMAN_SPECIFIC_HEAD],
            head_classes[HUMAN_SPECIFIC_HEAD],
            seed=args.seed + 101,
            max_per_class=max_per_human,
        )
        val_idx, human_val_balance = append_balanced_masked_head_indices(
            val_idx,
            balance_source_val_idx,
            labels[HUMAN_SPECIFIC_HEAD],
            sample_weight_masks[HUMAN_SPECIFIC_HEAD],
            head_classes[HUMAN_SPECIFIC_HEAD],
            seed=args.seed + 102,
            max_per_class=max_per_human,
        )
        test_idx, human_test_balance = append_balanced_masked_head_indices(
            test_idx,
            balance_source_test_idx,
            labels[HUMAN_SPECIFIC_HEAD],
            sample_weight_masks[HUMAN_SPECIFIC_HEAD],
            head_classes[HUMAN_SPECIFIC_HEAD],
            seed=args.seed + 103,
            max_per_class=max_per_human,
        )
        element_balance["human_specific_element_head"] = {
            "train": human_train_balance,
            "val": human_val_balance,
            "test": human_test_balance,
            "max_balanced_per_human_specific_element": int(max_per_human) if max_per_human else None,
        }
    if cds_detector_schema is not None and cds_only_mask is not None and (
        args.balance_elements or args.balance_organism_element or args.balance_kingdom_element
    ):
        max_per_cds = args.max_balanced_per_human_specific_element if args.max_balanced_per_human_specific_element > 0 else None
        cds_balance_info: dict[str, Any] = {}
        for split_name, split_ref, source_ref, seed_off in [
            ("train", "train_idx", "balance_source_train_idx", 201),
            ("val",   "val_idx",   "balance_source_val_idx",   202),
            ("test",  "test_idx",  "balance_source_test_idx",  203),
        ]:
            base = locals()[split_ref]
            source = locals()[source_ref]
            rng_cds = np.random.default_rng(args.seed + seed_off)
            cds_in_source = source[np.asarray(cds_only_mask, dtype=np.float32)[source] > 0]
            if max_per_cds is not None and len(cds_in_source) > max_per_cds:
                cds_in_source = rng_cds.choice(cds_in_source, size=max_per_cds, replace=False)
            combined = np.unique(np.concatenate([base, cds_in_source])).astype(np.int64)
            rng_cds.shuffle(combined)
            if split_name == "train":
                train_idx = combined
            elif split_name == "val":
                val_idx = combined
            else:
                test_idx = combined
            cds_balance_info[split_name] = {"n_cds_appended": int(len(cds_in_source)), "n_after": int(len(combined))}
        element_balance["cds_detector_head"] = cds_balance_info
    n_features = int(X_raw.shape[1])
    n_positions = int(X_raw.shape[2])
    table_window_masks = None
    table_window_counts = None
    table_window_defs: list[dict[str, Any]] = []
    if args.table_window_branch:
        table_window_masks, table_window_counts, table_window_defs = build_table_window_masks(
            parameter_to_idx=encoders["parameter"],
            n_positions=n_positions,
            original_n_positions=original_n_positions,
            profile_offset=profile_offset,
            table_center=args.table_center,
        )
        save_json(args.out / "table_window_definitions.json", {"windows": table_window_defs})
    parameter_channel_weights = None
    if args.paper_parameter_scaling:
        parameter_channel_weights = build_parameter_channel_weights(encoders["parameter"])
    start_stop_parameter_indices = [
        int(encoders["parameter"][name])
        for name in ("hbond", "stack", "sol", "bp", "intra", "bbone")
        if name in encoders["parameter"]
    ]
    profile_normalizer = None
    organism_normalization_info: dict[str, Any] = {"enabled": False}
    if args.organism_profile_normalization:
        print("Computing organism-conditioned profile normalization from training split...")
        _heldout_norm_idx = test_idx if getattr(args, "normalize_heldout_from_test", False) else None
        if _heldout_norm_idx is not None:
            print("  (held-out organisms will be normalized from their OWN test profiles — transductive, no label leak)")
        profile_normalizer = compute_organism_profile_normalizer(
            X_raw=X_raw,
            organism_labels=labels["organism"],
            train_idx=train_idx,
            n_organisms=head_classes["organism"],
            eps=float(args.organism_normalization_eps),
            chunk_size=int(args.organism_normalization_chunk_size),
            heldout_idx=_heldout_norm_idx,
        )
        normalizer_path = args.out / "organism_profile_normalization.npz"
        save_organism_profile_normalizer(
            normalizer_path,
            profile_normalizer,
            active_label_names["organism"],
        )
        organism_normalization_info = {
            "enabled": True,
            "path": str(normalizer_path),
            "basis": "train_split_after_scope_element_filtering_and_balancing",
            "eps": float(args.organism_normalization_eps),
            "chunk_size": int(args.organism_normalization_chunk_size),
            "counts": {
                name: int(profile_normalizer.counts[i])
                for i, name in enumerate(active_label_names["organism"])
            },
        }

    print(f"X_raw shape:          {X_raw.shape}")
    print(f"Original positions:   {original_n_positions}")
    print(f"Profile offset:       {profile_offset}")
    print(f"Training scope:       {args.scope} {args.scope_name or ''}".rstrip())
    print(f"Element set:          {args.element_set}")
    print(f"Train/val/test sizes: {len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,}")
    if args.balance_kingdom_element:
        print("Kingdom-element-balanced sampling:")
        print(f"  train total: {element_balance['train']['n_after']:,}")
        print(f"  val total:   {element_balance['val']['n_after']:,}")
        print(f"  test total:  {element_balance['test']['n_after']:,}")
    elif args.balance_organism_element:
        print("Organism-element-balanced sampling:")
        print(f"  train total: {element_balance['train']['n_after']:,}")
        print(f"  val total:   {element_balance['val']['n_after']:,}")
        print(f"  test total:  {element_balance['test']['n_after']:,}")
    elif args.balance_elements:
        print("Element-balanced sampling:")
        print(f"  train per element: {element_balance['train']['balanced_per_element']:,}")
        print(f"  val per element:   {element_balance['val']['balanced_per_element']:,}")
        print(f"  test per element:  {element_balance['test']['balanced_per_element']:,}")
    print("Heads:")
    for head, n_classes in head_classes.items():
        print(f"  {head:<8} {n_classes} classes")
    if combo_schema is not None:
        print(f"Auxiliary organism::element classes: {combo_schema['n_classes']}")
    if coarse_schema is not None:
        print(f"Auxiliary coarse element classes: {coarse_schema['n_classes']}")
    if bio_group_schema is not None:
        print(f"Biological group classes: {bio_group_schema['n_classes']}")
    if pair_schemas:
        print("Pair-specific heads:")
        for head, schema_part in pair_schemas.items():
            pair_labels = [item["label"] for item in schema_part["labels"]]
            print(f"  {head}: {pair_labels}")
    if boundary_context_schema is not None:
        boundary_labels = [item["label"] for item in boundary_context_schema["labels"]]
        print(f"Boundary-context head: {boundary_labels}")
    if human_specific_schema is not None:
        human_labels = [item["label"] for item in human_specific_schema["labels"]]
        print(f"Human-specific element head: {human_labels}")
        print(f"  masked samples: {human_specific_schema['n_masked_samples']:,}")
    print("Element scope:")
    print(f"  active:         {element_names}")
    print(f"  universal:      {schema['heads']['element']['universal_elements']}")
    print(f"  human-specific: {schema['heads']['element']['human_specific_elements']}")
    print("Organism-conditioned profile normalization:")
    print(f"  enabled: {bool(args.organism_profile_normalization)}")
    if profile_normalizer is not None:
        print(f"  counts:  {organism_normalization_info['counts']}")
    print("Table window branch:")
    print(f"  enabled: {bool(args.table_window_branch)}")
    print(f"  windows: {len(table_window_defs)}")
    print("Complexity feature branch:")
    print(f"  enabled: {bool(args.complexity_feature_branch)}")
    print("Profile capsule branch:")
    print(f"  enabled: {bool(args.profile_capsule_branch)}")
    if args.profile_capsule_branch:
        print(f"  capsules: {args.profile_capsule_num_capsules}")
        print(f"  dim:      {args.profile_capsule_dim}")
    print("Paper-guided start/stop features:")
    print(f"  parameter scaling: {bool(args.paper_parameter_scaling)}")
    print(f"  contrast branch:   {bool(args.start_stop_contrast_branch)}")
    print(f"  contrast params:   {[name for name in ('hbond', 'stack', 'sol', 'bp', 'intra', 'bbone') if name in encoders['parameter']]}")
    print("Sequence motif branch:")
    print(f"  enabled: {bool(args.sequence_branch)}")
    print("Explicit motif detection branch:")
    print(f"  enabled: {bool(args.motif_detection_branch)}")

    sequence_accessor = None
    if args.sequence_branch or args.motif_detection_branch:
        manifest_path = args.sequence_manifest
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Sequence manifest not found: {manifest_path}")
        sequence_accessor = SequenceManifestAccessor(
            manifest_path=manifest_path,
            sequence_root=args.sequence_root,
            sequence_length=args.sequence_length,
            preload=args.preload_sequences,
        )
        print(f"  manifest: {manifest_path}")
        print(f"  length:   {args.sequence_length}")
    kg_feature_provider = None
    if args.kg_branch:
        kg_dir = args.kg_dir
        if not kg_dir.is_absolute():
            kg_dir = Path.cwd() / kg_dir
        kg_feature_provider = KGFeatureProvider(
            kg_dir=kg_dir,
            labels=labels,
            active_label_names=active_label_names,
        )
        print("Knowledge graph branch:")
        print(f"  enabled: true")
        print(f"  kg_dir:  {kg_dir}")
        print(f"  feature_dim: {kg_feature_provider.feature_dim}")
    else:
        print("Knowledge graph branch:")
        print("  enabled: false")

    train_class_weights = None
    if not args.no_class_weights:
        train_class_weights = {}
        for head, values in labels.items():
            if head in sample_weight_masks:
                valid = sample_weight_masks[head][train_idx] > 0
                train_class_weights[head] = make_class_weight_vector(
                    values[train_idx][valid],
                    head_classes[head],
                )
            else:
                train_class_weights[head] = make_class_weight_vector(
                    values[train_idx],
                    head_classes[head],
                )
        print("Class weights:")
        for head, weights in train_class_weights.items():
            print(f"  {head}: {[round(float(w), 3) for w in weights]}")

    # Remove global organism/element heads from head_classes now that balancing
    # is done. labels["organism"] is kept for the profile normalizer.
    if args.kingdom_specific_organism_head:
        del head_classes["organism"]
    if args.kingdom_specific_element_head:
        del head_classes["element"]

    # When kingdom-specific heads are used, labels["organism"/"element"] are
    # kept for the profile normalizer / eval but must not appear in y for Keras.
    seq_output_heads = (
        set(head_classes.keys())
        if (args.kingdom_specific_organism_head or args.kingdom_specific_element_head)
        else None
    )

    train_seq = MultiHeadProfileSequence(
        X_raw, labels, train_idx, args.batch_size,
        class_weights=train_class_weights, sample_weight_masks=sample_weight_masks,
        sequence_accessor=sequence_accessor,
        kg_feature_provider=kg_feature_provider,
        profile_normalizer=profile_normalizer,
        kingdom_label_conditioning=args.kingdom_label_conditioning,
        n_kingdom_classes=head_classes["kingdom"],
        shuffle=True, seed=args.seed,
        output_heads=seq_output_heads,
    )
    val_seq = MultiHeadProfileSequence(
        X_raw, labels, val_idx, args.batch_size,
        class_weights=None, sample_weight_masks=sample_weight_masks,
        sequence_accessor=sequence_accessor,
        kg_feature_provider=kg_feature_provider,
        profile_normalizer=profile_normalizer,
        kingdom_label_conditioning=args.kingdom_label_conditioning,
        n_kingdom_classes=head_classes["kingdom"],
        shuffle=False, seed=args.seed + 1,
        output_heads=seq_output_heads,
    )
    test_seq = MultiHeadProfileSequence(
        X_raw, labels, test_idx, args.batch_size,
        class_weights=None, sample_weight_masks=sample_weight_masks,
        sequence_accessor=sequence_accessor,
        kg_feature_provider=kg_feature_provider,
        profile_normalizer=profile_normalizer,
        kingdom_label_conditioning=args.kingdom_label_conditioning,
        n_kingdom_classes=head_classes["kingdom"],
        shuffle=False, seed=args.seed + 2,
        output_heads=seq_output_heads,
    )

    gpus = tf.config.list_physical_devices("GPU")
    print(f"Visible GPUs: {len(gpus)}")
    for g in gpus:
        print(f"  {g.name}  details={tf.config.experimental.get_device_details(g)}")
    strategy = tf.distribute.MirroredStrategy()
    print(f"MirroredStrategy: {strategy.num_replicas_in_sync} GPU(s)")
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    print("Mixed precision: mixed_bfloat16")

    ckpt_latest = ckpt_dir / "latest_model.keras"
    initial_epoch = 0

    loss_weights = {
        "kingdom": float(args.kingdom_loss_weight),
    }
    if args.kingdom_specific_element_head:
        for k_name in kingdom_element_head_classes:
            loss_weights[f"element_{k_name}"] = float(args.element_loss_weight)
    else:
        loss_weights["element"] = float(args.element_loss_weight)
    if args.kingdom_specific_organism_head:
        for k_name in kingdom_organism_head_classes:
            loss_weights[f"organism_{k_name}"] = float(args.organism_loss_weight)
    else:
        loss_weights["organism"] = float(args.organism_loss_weight)
    if HUMAN_SPECIFIC_HEAD in head_classes:
        loss_weights[HUMAN_SPECIFIC_HEAD] = float(args.human_specific_element_loss_weight)
    if UTR_SUBTYPE_HEAD in head_classes:
        loss_weights[UTR_SUBTYPE_HEAD] = float(args.utr_subtype_loss_weight)
    if CDS_DETECTOR_HEAD in head_classes:
        loss_weights[CDS_DETECTOR_HEAD] = float(args.cds_detector_loss_weight)
    if "aux_organism_element" in head_classes:
        loss_weights["aux_organism_element"] = float(args.aux_organism_element_loss_weight)
    if "aux_element_coarse" in head_classes:
        loss_weights["aux_element_coarse"] = float(args.aux_element_coarse_loss_weight)
    if "bio_element_group" in head_classes:
        loss_weights["bio_element_group"] = float(args.bio_element_group_loss_weight)
    for pair_head in PAIR_ELEMENT_HEADS:
        if pair_head in head_classes:
            loss_weights[pair_head] = float(args.pair_element_loss_weight)
    if BOUNDARY_CONTEXT_HEAD in head_classes:
        loss_weights[BOUNDARY_CONTEXT_HEAD] = float(args.boundary_context_loss_weight)

    losses: dict[str, Any] = {head: "sparse_categorical_crossentropy" for head in head_classes}
    if float(args.element_focal_loss_gamma) > 0.0:
        focal = SparseCategoricalFocalLoss(
            gamma=float(args.element_focal_loss_gamma),
            label_smoothing=float(args.element_label_smoothing),
        )
        if args.kingdom_specific_element_head:
            for k_name in kingdom_element_head_classes:
                losses[f"element_{k_name}"] = focal
        else:
            losses["element"] = focal
        print(f"Element head loss: sparse categorical focal loss, gamma={args.element_focal_loss_gamma:g}, label_smoothing={args.element_label_smoothing:g}")
    else:
        print("Element head loss: sparse categorical cross-entropy")

    if args.resume and ckpt_latest.exists():
        print(f"Resuming: loading checkpoint {ckpt_latest}")
        with strategy.scope():
            model = tf.keras.models.load_model(str(ckpt_latest))
        epoch_file = ckpt_dir / "last_epoch.txt"
        if epoch_file.exists():
            initial_epoch = int(epoch_file.read_text().strip()) + 1
        print(f"Continuing from epoch {initial_epoch}/{args.epochs}")
    else:
        if args.resume:
            print(f"Warning: --resume set but no checkpoint found at {ckpt_latest}, starting from scratch")
        with strategy.scope():
            model = build_model(
                n_positions=n_positions,
                n_features=n_features,
                head_classes=head_classes,
                d_model=args.d_model,
                n_stages=args.n_stages,
                blocks_per_stage=args.blocks_per_stage,
                stem_kernel=args.stem_kernel,
                block_kernel=args.block_kernel,
                dropout=args.dropout,
                table_window_masks=table_window_masks,
                table_window_counts=table_window_counts,
                use_table_windows=args.table_window_branch,
                use_complexity_features=args.complexity_feature_branch,
                use_profile_capsule_branch=args.profile_capsule_branch,
                profile_capsule_num_capsules=args.profile_capsule_num_capsules,
                profile_capsule_dim=args.profile_capsule_dim,
                use_sequence_branch=args.sequence_branch,
                use_motif_detection_branch=args.motif_detection_branch,
                motif_detection_dim=args.motif_detection_dim,
                sequence_length=args.sequence_length,
                use_kg_branch=args.kg_branch,
                kg_feature_dim=kg_feature_provider.feature_dim if kg_feature_provider is not None else 0,
                legacy_kingdom_conditioning=args.legacy_kingdom_conditioning,
                kingdom_label_conditioning=args.kingdom_label_conditioning,
                kingdom_gradient_reversal_lambda=args.kingdom_gradient_reversal_lambda,
                condition_element_on_organism=args.condition_element_on_organism,
                organism_gradient_reversal_lambda=args.organism_gradient_reversal_lambda,
                parameter_channel_weights=parameter_channel_weights,
                start_stop_parameter_indices=start_stop_parameter_indices,
                use_start_stop_contrast=args.start_stop_contrast_branch,
                kingdom_specific_organism_head=args.kingdom_specific_organism_head,
                kingdom_organism_head_classes=kingdom_organism_head_classes if args.kingdom_specific_organism_head else None,
                kingdom_specific_element_head=args.kingdom_specific_element_head,
                kingdom_element_head_classes=kingdom_element_head_classes if args.kingdom_specific_element_head else None,
            )
            if float(args.weight_decay) > 0.0:
                optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=args.lr, weight_decay=float(args.weight_decay), clipnorm=1.0
                )
                print(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
                print(f"Optimizer: Adam (lr={args.lr})")
            model.compile(
                optimizer=optimizer,
                loss=losses,
                loss_weights=loss_weights,
                # NOTE: MaskedElementAccuracy must live in `weighted_metrics`, NOT
                # `metrics`. Keras only routes sample_weight to update_state for
                # weighted_metrics; a metric in plain `metrics` is always called
                # with sample_weight=None, so MaskedElementAccuracy fell into its
                # unweighted else-branch and counted the ~74k zero-weight CDS/
                # human-specific dummy rows as predictions — re-introducing the
                # exact dilution artifact (~0.61) it was written to remove. The
                # honest filtered accuracy is ~0.72 (see element_per_class_report).
                metrics={
                    head: ["accuracy"] for head in head_classes if head != "element"
                },
                weighted_metrics={
                    "element": [MaskedElementAccuracy(name="masked_accuracy")],
                },
                jit_compile=False,
                run_eagerly=False,
            )

    def _grl_warmup(epoch, logs=None):
        # Ramp GRL lambda 0 -> target over the first grl_warmup_epochs epochs so the
        # adversarial heads don't blow up early (needed for the no-sequence arms).
        w = getattr(args, "grl_warmup_epochs", 0) or 0
        if w <= 0:
            return
        frac = min(1.0, float(epoch) / float(w))
        for nm in ("kingdom_gradient_reversal", "organism_gradient_reversal"):
            try:
                lyr = model.get_layer(nm)
            except ValueError:
                continue
            if hasattr(lyr, "lambda_var"):
                lyr.lambda_var.assign(frac * lyr.target_lambda)
                print(f"[GRL warmup] epoch {epoch}: {nm} lambda -> {frac * lyr.target_lambda:.4f}", flush=True)

    callbacks = [
        tf.keras.callbacks.LambdaCallback(on_epoch_begin=_grl_warmup),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best_model.keras"),
            monitor="val_element_masked_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_element_masked_accuracy",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_element_masked_accuracy",
            mode="max",
            factor=0.5,
            patience=max(2, args.patience // 3),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(args.out / "history.csv"), append=args.resume),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_latest),
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: (ckpt_dir / "last_epoch.txt").write_text(str(epoch))
        ),
    ]

    save_json(
        args.out / "run_config.json",
        {
            "dataset": str(args.dataset),
            "out": str(args.out),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "scope": scope_info,
            "d_model": int(args.d_model),
            "n_stages": int(args.n_stages),
            "blocks_per_stage": int(args.blocks_per_stage),
            "stem_kernel": int(args.stem_kernel),
            "block_kernel": int(args.block_kernel),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "crop_window": int(args.crop_window),
            "profile_offset": int(profile_offset),
            "original_n_positions": int(original_n_positions),
            "organism_profile_normalization": organism_normalization_info,
            "table_window_branch": {
                "enabled": bool(args.table_window_branch),
                "n_windows": int(len(table_window_defs)),
                "table_center": int(args.table_center),
                "definitions_path": str(args.out / "table_window_definitions.json"),
            },
            "complexity_feature_branch": {
                "enabled": bool(args.complexity_feature_branch),
            },
            "profile_capsule_branch": {
                "enabled": bool(args.profile_capsule_branch),
                "num_capsules": int(args.profile_capsule_num_capsules),
                "capsule_dim": int(args.profile_capsule_dim),
                "branch_type": "capsule_style_position_weighted_profile_feature_pooling",
            },
            "sequence_branch": {
                "enabled": bool(args.sequence_branch),
                "manifest": str(args.sequence_manifest),
                "sequence_root": str(args.sequence_root) if args.sequence_root is not None else None,
                "sequence_length": int(args.sequence_length),
                "branch_type": "one_hot_cnn_motif_encoder",
            },
            "motif_detection_branch": {
                "enabled": bool(args.motif_detection_branch),
                "projection_dim": int(args.motif_detection_dim),
                "motifs": ExplicitMotifFeatures.DEFAULT_MOTIFS if tf is not None else [],
                "branch_type": "explicit_position_aware_motif_and_composition_features",
            },
            "kg_branch": {
                "enabled": bool(args.kg_branch),
                "kg_dir": str(args.kg_dir),
                "feature_dim": int(kg_feature_provider.feature_dim) if kg_feature_provider is not None else 0,
                "branch_type": "context_only_kg_node_feature_embedding",
                "uses_target_element_node_as_input": False,
            },
            "legacy_kingdom_conditioning": {
                "enabled": bool(args.legacy_kingdom_conditioning),
                "basis": "May-19 best run conditioned organism and element towers on kingdom_repr even under one-class kingdom scope.",
            },
            "kingdom_label_conditioning": {
                "enabled": bool(args.kingdom_label_conditioning),
                "basis": "Ground-truth kingdom one-hot injected as explicit element tower input. "
                         "Kingdom is known from organism metadata at inference time — not label leakage.",
            },
            "kingdom_gradient_reversal": {
                "enabled": args.kingdom_gradient_reversal_lambda > 0,
                "lambda": float(args.kingdom_gradient_reversal_lambda),
                "basis": "Adversarial kingdom head forces shared representation to suppress kingdom-predictive "
                         "features. Pairs with kingdom_label_conditioning so element tower still receives "
                         "kingdom identity via ground-truth input rather than from shared features.",
            },
            "element_conditioning": {
                "condition_on_organism_repr": bool(args.condition_element_on_organism),
                "condition_on_kingdom_repr": bool(
                    args.kingdom_gradient_reversal_lambda <= 0
                    and (args.legacy_kingdom_conditioning or head_classes["kingdom"] > 1)
                ),
                "condition_on_kingdom_label": bool(args.kingdom_label_conditioning),
                "organism_gradient_reversal_lambda": float(args.organism_gradient_reversal_lambda),
                "defensibility_note": (
                    "When condition_on_organism_repr is false, organism is predicted as an auxiliary task "
                    "but its tower representation is not directly passed into the element classifier. "
                    "When organism_gradient_reversal_lambda is >0, the organism head is adversarial "
                    "with respect to the shared representation."
                ),
            },
            "element_label_collapse": element_collapse_info,
            "human_specific_element_head": {
                "enabled": human_specific_schema is not None,
                "loss_weight": float(args.human_specific_element_loss_weight),
                "n_classes": int(human_specific_schema["n_classes"]) if human_specific_schema is not None else 0,
                "labels": [item["label"] for item in human_specific_schema["labels"]] if human_specific_schema is not None else [],
                "schema_path": str(args.out / "human_specific_element_label_schema.json") if human_specific_schema is not None else None,
                "masked_training": True,
                "removed_from_main_element_head": (
                    human_specific_schema["removed_from_main_element_head"]
                    if human_specific_schema is not None
                    else []
                ),
            },
            "paper_guided_start_stop": {
                "parameter_scaling": bool(args.paper_parameter_scaling),
                "parameter_weights": START_STOP_PARAMETER_WEIGHTS,
                "contrast_branch": bool(args.start_stop_contrast_branch),
                "contrast_parameters": [
                    name for name in ("hbond", "stack", "sol", "bp", "intra", "bbone")
                    if name in encoders["parameter"]
                ],
                "paper_basis": (
                    "Start codon H-bond and stacking energies decline, stop codon H-bond increases, "
                    "stop stacking has a less pronounced dip; solvation, BP-axis, intra, and backbone "
                    "show start/stop shifts."
                ),
            },
            "element_balance": element_balance,
            "loss_weights": loss_weights,
            "losses": {
                head: (
                    {
                        "name": "sparse_categorical_focal_loss",
                        "gamma": float(args.element_focal_loss_gamma),
                    }
                    if head == "element" and float(args.element_focal_loss_gamma) > 0.0
                    else {"name": "sparse_categorical_crossentropy"}
                )
                for head in head_classes
            },
            "aux_organism_element_head": {
                "enabled": bool(args.aux_organism_element_head),
                "n_classes": int(combo_schema["n_classes"]) if combo_schema is not None else 0,
                "schema_path": str(args.out / "organism_element_label_schema.json") if combo_schema is not None else None,
            },
            "aux_element_coarse_head": {
                "enabled": bool(args.aux_element_coarse_head),
                "n_classes": int(coarse_schema["n_classes"]) if coarse_schema is not None else 0,
                "schema_path": str(args.out / "coarse_element_label_schema.json") if coarse_schema is not None else None,
            },
            "bio_hierarchy_heads": {
                "enabled": bool(args.bio_hierarchy_heads),
                "bio_group_classes": int(bio_group_schema["n_classes"]) if bio_group_schema is not None else 0,
                "bio_group_schema_path": str(args.out / "bio_element_group_label_schema.json") if bio_group_schema is not None else None,
                "pair_heads": sorted(pair_schemas),
                "pair_schema_path": str(args.out / "pair_element_label_schema.json") if pair_schemas else None,
            },
            "boundary_context_head": {
                "enabled": boundary_context_schema is not None,
                "classes": int(boundary_context_schema["n_classes"]) if boundary_context_schema is not None else 0,
                "members": [item["label"] for item in boundary_context_schema["labels"]] if boundary_context_schema is not None else [],
                "schema_path": str(args.out / "boundary_context_label_schema.json") if boundary_context_schema is not None else None,
            },
            "head_classes": head_classes,
            "n_features": n_features,
            "n_positions": n_positions,
            "head_architecture": "taxonomy_conditioned_element_with_biological_hierarchy_heads",
            **RUNTIME_CONFIG,
        },
    )

    model.summary(line_length=110)

    n_replicas = strategy.num_replicas_in_sync
    # __len__ uses floor division so every batch is exactly batch_size samples.
    # steps_per_epoch == len(seq) ensures the iterator fully exhausts each epoch
    # and on_epoch_end() fires correctly for shuffling.
    steps_per_epoch = len(train_seq)
    validation_steps = len(val_seq)

    print(f"steps_per_epoch={steps_per_epoch}  validation_steps={validation_steps}  replicas={n_replicas}")

    if args.eval_only:
        ckpt = Path(args.load_model) if args.load_model else (args.out / "checkpoints" / "best_model.keras")
        print(f"EVAL-ONLY: loading weights from {ckpt} (skipping training)")
        model.load_weights(str(ckpt))
        history = type("H", (), {"history": {}})()
    else:
        history = model.fit(
            train_seq,
            validation_data=val_seq,
            initial_epoch=initial_epoch,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )
        write_history_csv(args.out / "history_final.csv", history.history)

    print("\nEvaluating on test split...")
    test_results = model.evaluate(test_seq, verbose=1, return_dict=True)
    test_results = {k: float(v) for k, v in test_results.items()}

    print("\nWriting per-class element diagnostics...")
    pred_outputs = model.predict(test_seq, verbose=1)
    output_names = list(model.output_names)

    def get_prob(head: str):
        if isinstance(pred_outputs, dict):
            return pred_outputs[head]
        return pred_outputs[output_names.index(head)]

    test_iteration_idx = test_seq.iteration_indices()
    if args.kingdom_specific_element_head and kingdom_to_global_elem_ids:
        kingdom_true_test = labels["kingdom"][test_iteration_idx].astype(np.int32)
        element_pred = np.zeros(len(test_iteration_idx), dtype=np.int32)
        for k_id, k_name in enumerate(active_label_names["kingdom"]):
            k_mask = kingdom_true_test == k_id
            if not np.any(k_mask):
                continue
            elem_prob_k = get_prob(f"element_{k_name}")
            local_pred = np.argmax(elem_prob_k[k_mask], axis=1)
            global_ids = np.array(kingdom_to_global_elem_ids[k_name], dtype=np.int32)
            element_pred[k_mask] = global_ids[local_pred]
        global_elem_acc = float(np.mean(element_pred == labels["element"][test_iteration_idx]))
        test_results["reconstructed_element_accuracy"] = global_elem_acc
        print(f"Reconstructed global element accuracy: {global_elem_acc:.4f}")
    else:
        element_prob = get_prob("element")
        element_pred = np.argmax(element_prob, axis=1).astype(np.int32)
    element_true = labels["element"][test_iteration_idx].astype(np.int32)
    element_labels = active_label_names["element"]
    # Filter to samples with active element mask so masked-out samples (e.g.
    # CDS-appended dummy labels) don't inflate per-class support counts.
    # element_valid is reused below to keep the hierarchical-fusion accuracies
    # clean-vs-clean (the ~74k zero-weight dummy rows are all true-class-0 and
    # otherwise dilute every element-head accuracy down to the ~0.61 artifact).
    if "element" in sample_weight_masks:
        element_valid = sample_weight_masks["element"][test_iteration_idx] > 0
    else:
        element_valid = np.ones(len(element_true), dtype=bool)
    element_true_r = element_true[element_valid]
    element_pred_r = element_pred[element_valid]
    write_per_class_report(
        args.out / "element_per_class_report.csv",
        element_true_r,
        element_pred_r,
        element_labels,
    )

    # ---- per-kingdom & per-organism masked element accuracy (universality table) ----
    king_r = labels["kingdom"][test_iteration_idx][element_valid].astype(np.int32)
    org_r = labels["organism"][test_iteration_idx][element_valid].astype(np.int32)
    elem_correct = (element_pred_r == element_true_r)
    per_kingdom = {}
    for kid, kname in enumerate(active_label_names["kingdom"]):
        m = king_r == kid
        if m.any():
            per_kingdom[kname] = {"element_masked_accuracy": round(float(elem_correct[m].mean()), 4),
                                  "support": int(m.sum())}
    per_organism = {}
    for oid, oname in enumerate(active_label_names["organism"]):
        m = org_r == oid
        if m.any():
            per_organism[oname] = {"element_masked_accuracy": round(float(elem_correct[m].mean()), 4),
                                   "support": int(m.sum())}
    save_json(args.out / "per_kingdom_element_accuracy.json",
              {"per_kingdom": per_kingdom, "per_organism": per_organism})
    print("\nPer-kingdom element_masked_accuracy (test):")
    for k, v in per_kingdom.items():
        print(f"  {k:10s} {v['element_masked_accuracy']:.4f}  (n={v['support']:,})")
    write_confusion_csv(
        args.out / "element_confusion_matrix.csv",
        element_true_r,
        element_pred_r,
        element_labels,
    )
    if args.kg_branch:
        kg_dir = args.kg_dir if args.kg_dir.is_absolute() else Path.cwd() / args.kg_dir
        write_kg_posthoc_element_confusions(
            args.out / "kg_posthoc_element_confusions.csv",
            element_true,
            element_pred,
            element_labels,
            kg_dir=kg_dir,
            min_count=1,
        )

    # Reconstruct global organism accuracy from per-kingdom heads.
    if args.kingdom_specific_organism_head and kingdom_to_global_org_ids:
        kingdom_true_test = labels["kingdom"][test_iteration_idx].astype(np.int32)
        org_true_test = np.load(args.dataset / "Y_organism.npy")[test_iteration_idx].astype(np.int32)
        org_pred_global = np.zeros(len(test_iteration_idx), dtype=np.int32)
        for k_id, k_name in enumerate(active_label_names["kingdom"]):
            k_mask = kingdom_true_test == k_id
            if not np.any(k_mask):
                continue
            org_prob_k = get_prob(f"organism_{k_name}")
            local_pred = np.argmax(org_prob_k[k_mask], axis=1)
            global_ids = np.array(kingdom_to_global_org_ids[k_name], dtype=np.int32)
            org_pred_global[k_mask] = global_ids[local_pred]
        global_org_acc = float(np.mean(org_pred_global == org_true_test))
        test_results["reconstructed_organism_accuracy"] = global_org_acc
        print(f"Reconstructed global organism accuracy: {global_org_acc:.4f}")
        org_labels_all = active_label_names["organism"]
        write_per_class_report(
            args.out / "organism_per_class_report.csv",
            org_true_test, org_pred_global, org_labels_all,
        )
        write_confusion_csv(
            args.out / "organism_confusion_matrix.csv",
            org_true_test, org_pred_global, org_labels_all,
        )
        extra_report_paths_org = {
            "organism_per_class_report": str(args.out / "organism_per_class_report.csv"),
            "organism_confusion_matrix": str(args.out / "organism_confusion_matrix.csv"),
        }
    else:
        extra_report_paths_org = {}

    extra_report_paths: dict[str, str] = {**extra_report_paths_org}
    if args.kg_branch:
        extra_report_paths["kg_posthoc_element_confusions"] = str(
            args.out / "kg_posthoc_element_confusions.csv"
        )
    if HUMAN_SPECIFIC_HEAD in head_classes and human_specific_schema is not None:
        human_mask = sample_weight_masks[HUMAN_SPECIFIC_HEAD][test_iteration_idx] > 0
        if np.any(human_mask):
            human_prob = get_prob(HUMAN_SPECIFIC_HEAD)
            human_pred = np.argmax(human_prob, axis=1).astype(np.int32)[human_mask]
            human_true = labels[HUMAN_SPECIFIC_HEAD][test_iteration_idx].astype(np.int32)[human_mask]
            human_labels = [item["label"] for item in human_specific_schema["labels"]]
            human_accuracy = float(np.mean(human_pred == human_true))
            test_results["human_specific_element_masked_accuracy"] = human_accuracy
            test_results["human_specific_element_support"] = float(np.sum(human_mask))
            report_path = args.out / "human_specific_element_per_class_report.csv"
            confusion_path = args.out / "human_specific_element_confusion_matrix.csv"
            write_per_class_report(report_path, human_true, human_pred, human_labels)
            write_confusion_csv(confusion_path, human_true, human_pred, human_labels)
            extra_report_paths["human_specific_element_per_class_report"] = str(report_path)
            extra_report_paths["human_specific_element_confusion_matrix"] = str(confusion_path)
    bio_labels: list[str] | None = None
    bio_prob: np.ndarray | None = None
    if "bio_element_group" in head_classes and bio_group_schema is not None:
        bio_prob = get_prob("bio_element_group")
        bio_pred = np.argmax(bio_prob, axis=1).astype(np.int32)
        bio_true = labels["bio_element_group"][test_iteration_idx].astype(np.int32)
        bio_labels = [item["label"] for item in bio_group_schema["labels"]]
        write_per_class_report(
            args.out / "bio_element_group_per_class_report.csv",
            bio_true,
            bio_pred,
            bio_labels,
        )
        write_confusion_csv(
            args.out / "bio_element_group_confusion_matrix.csv",
            bio_true,
            bio_pred,
            bio_labels,
        )
        extra_report_paths["bio_element_group_per_class_report"] = str(
            args.out / "bio_element_group_per_class_report.csv"
        )
        extra_report_paths["bio_element_group_confusion_matrix"] = str(
            args.out / "bio_element_group_confusion_matrix.csv"
        )

    pair_probs: dict[str, np.ndarray] = {}
    for pair_head, pair_schema in pair_schemas.items():
        if pair_head not in head_classes:
            continue
        pair_mask = sample_weight_masks[pair_head][test_iteration_idx] > 0
        if not np.any(pair_mask):
            continue
        pair_prob = get_prob(pair_head)
        pair_probs[pair_head] = pair_prob
        pair_pred = np.argmax(pair_prob, axis=1).astype(np.int32)[pair_mask]
        pair_true = labels[pair_head][test_iteration_idx].astype(np.int32)[pair_mask]
        pair_labels = [item["label"] for item in pair_schema["labels"]]
        report_path = args.out / f"{pair_head}_per_class_report.csv"
        confusion_path = args.out / f"{pair_head}_confusion_matrix.csv"
        write_per_class_report(report_path, pair_true, pair_pred, pair_labels)
        write_confusion_csv(confusion_path, pair_true, pair_pred, pair_labels)
        extra_report_paths[f"{pair_head}_per_class_report"] = str(report_path)
        extra_report_paths[f"{pair_head}_confusion_matrix"] = str(confusion_path)

    if UTR_SUBTYPE_HEAD in head_classes and utr_subtype_schema is not None:
        utr_mask = sample_weight_masks[UTR_SUBTYPE_HEAD][test_iteration_idx] > 0
        if np.any(utr_mask):
            utr_prob = get_prob(UTR_SUBTYPE_HEAD)
            utr_pred = np.argmax(utr_prob, axis=1).astype(np.int32)[utr_mask]
            utr_true = labels[UTR_SUBTYPE_HEAD][test_iteration_idx].astype(np.int32)[utr_mask]
            utr_class_labels = [item["label"] for item in utr_subtype_schema["labels"]]
            report_path = args.out / "utr_subtype_per_class_report.csv"
            confusion_path = args.out / "utr_subtype_confusion_matrix.csv"
            write_per_class_report(report_path, utr_true, utr_pred, utr_class_labels)
            write_confusion_csv(confusion_path, utr_true, utr_pred, utr_class_labels)
            extra_report_paths["utr_subtype_per_class_report"] = str(report_path)
            extra_report_paths["utr_subtype_confusion_matrix"] = str(confusion_path)

    if CDS_DETECTOR_HEAD in head_classes and cds_detector_schema is not None:
        cds_det_prob = get_prob(CDS_DETECTOR_HEAD)
        cds_det_pred = np.argmax(cds_det_prob, axis=1).astype(np.int32)
        cds_det_true = labels[CDS_DETECTOR_HEAD][test_iteration_idx].astype(np.int32)
        cds_class_labels = ["non_cds", "cds"]
        cds_only_mask = cds_det_true == 1
        if np.any(cds_only_mask):
            test_results["cds_detector_cds_recall"] = float(
                np.mean(cds_det_pred[cds_only_mask] == 1)
            )
            test_results["cds_detector_support"] = float(np.sum(cds_only_mask))
        report_path = args.out / "cds_detector_per_class_report.csv"
        confusion_path = args.out / "cds_detector_confusion_matrix.csv"
        write_per_class_report(report_path, cds_det_true, cds_det_pred, cds_class_labels)
        write_confusion_csv(confusion_path, cds_det_true, cds_det_pred, cds_class_labels)
        extra_report_paths["cds_detector_per_class_report"] = str(report_path)
        extra_report_paths["cds_detector_confusion_matrix"] = str(confusion_path)

    if BOUNDARY_CONTEXT_HEAD in head_classes and boundary_context_schema is not None:
        boundary_mask = sample_weight_masks[BOUNDARY_CONTEXT_HEAD][test_iteration_idx] > 0
        if np.any(boundary_mask):
            boundary_prob = get_prob(BOUNDARY_CONTEXT_HEAD)
            boundary_pred = np.argmax(boundary_prob, axis=1).astype(np.int32)[boundary_mask]
            boundary_true = labels[BOUNDARY_CONTEXT_HEAD][test_iteration_idx].astype(np.int32)[boundary_mask]
            boundary_labels = [item["label"] for item in boundary_context_schema["labels"]]
            report_path = args.out / "boundary_context_per_class_report.csv"
            confusion_path = args.out / "boundary_context_confusion_matrix.csv"
            write_per_class_report(report_path, boundary_true, boundary_pred, boundary_labels)
            write_confusion_csv(confusion_path, boundary_true, boundary_pred, boundary_labels)
            extra_report_paths["boundary_context_per_class_report"] = str(report_path)
            extra_report_paths["boundary_context_confusion_matrix"] = str(confusion_path)

    fused_info: dict[str, Any] | None = None
    if not args.kingdom_specific_element_head and bio_prob is not None and bio_labels is not None and "pair_stac_stoc" in pair_probs:
        fused_pred, fused_info = make_hierarchical_fused_element_predictions(
            element_prob=element_prob,
            bio_prob=bio_prob,
            element_labels=element_labels,
            bio_labels=bio_labels,
            pair_probs=pair_probs,
            pair_schemas=pair_schemas,
        )
        # Clean (main-element only): exclude zero-weight dummy rows so the fusion
        # number is comparable to the filtered element accuracy, not the artifact.
        fused_pred_r = fused_pred[element_valid]
        fused_accuracy = float(np.mean(fused_pred_r == element_true_r))
        test_results["hierarchical_fused_element_accuracy"] = fused_accuracy
        write_per_class_report(
            args.out / "hierarchical_fused_element_per_class_report.csv",
            element_true_r,
            fused_pred_r,
            element_labels,
        )
        write_confusion_csv(
            args.out / "hierarchical_fused_element_confusion_matrix.csv",
            element_true_r,
            fused_pred_r,
            element_labels,
        )
        fused_info["accuracy"] = fused_accuracy
        extra_report_paths["hierarchical_fused_element_per_class_report"] = str(
            args.out / "hierarchical_fused_element_per_class_report.csv"
        )
        extra_report_paths["hierarchical_fused_element_confusion_matrix"] = str(
            args.out / "hierarchical_fused_element_confusion_matrix.csv"
        )

    soft_fused_info: dict[str, Any] | None = None
    if not args.kingdom_specific_element_head and bio_prob is not None and bio_labels is not None and "pair_stac_stoc" in pair_probs:
        print("\nCalibrating hierarchical soft fusion on validation split...")
        val_outputs = model.predict(val_seq, verbose=1)
        val_output_names = list(model.output_names)

        def get_val_prob(head: str):
            if isinstance(val_outputs, dict):
                return val_outputs[head]
            return val_outputs[val_output_names.index(head)]

        val_element_prob = get_val_prob("element")
        val_bio_prob = get_val_prob("bio_element_group")
        val_pair_probs = {
            pair_head: get_val_prob(pair_head)
            for pair_head in pair_schemas
            if pair_head in head_classes
        }
        val_hierarchy_prob = make_hierarchical_soft_element_probabilities(
            element_prob=val_element_prob,
            bio_prob=val_bio_prob,
            element_labels=element_labels,
            bio_labels=bio_labels,
            pair_probs=val_pair_probs,
            pair_schemas=pair_schemas,
        )
        val_iteration_idx = val_seq.iteration_indices()
        val_element_true = labels["element"][val_iteration_idx].astype(np.int32)
        # Tune alpha on main-element val samples only — otherwise the dummy rows
        # (true-class-0) bias alpha toward whatever shifts mass onto exon_boundary.
        if "element" in sample_weight_masks:
            val_element_valid = sample_weight_masks["element"][val_iteration_idx] > 0
        else:
            val_element_valid = np.ones(len(val_element_true), dtype=bool)
        soft_calibration = tune_hierarchical_soft_alpha(
            element_prob=val_element_prob[val_element_valid],
            hierarchy_prob=val_hierarchy_prob[val_element_valid],
            y_true=val_element_true[val_element_valid],
        )

        test_hierarchy_prob = make_hierarchical_soft_element_probabilities(
            element_prob=element_prob,
            bio_prob=bio_prob,
            element_labels=element_labels,
            bio_labels=bio_labels,
            pair_probs=pair_probs,
            pair_schemas=pair_schemas,
        )
        soft_prob = blend_element_probabilities(
            element_prob=element_prob,
            hierarchy_prob=test_hierarchy_prob,
            alpha=float(soft_calibration["best_alpha"]),
        )
        soft_pred = np.argmax(soft_prob, axis=1).astype(np.int32)
        # Clean (main-element only) — same filter as element/hard-fused above.
        soft_pred_r = soft_pred[element_valid]
        soft_accuracy = float(np.mean(soft_pred_r == element_true_r))
        test_results["hierarchical_soft_fused_element_accuracy"] = soft_accuracy
        write_per_class_report(
            args.out / "hierarchical_soft_fused_element_per_class_report.csv",
            element_true_r,
            soft_pred_r,
            element_labels,
        )
        write_confusion_csv(
            args.out / "hierarchical_soft_fused_element_confusion_matrix.csv",
            element_true_r,
            soft_pred_r,
            element_labels,
        )
        soft_fused_info = {
            "strategy": "validation_tuned_soft_blend_of_element_and_hierarchy_probabilities",
            "accuracy": soft_accuracy,
            **soft_calibration,
        }
        save_json(args.out / "hierarchical_soft_fused_element_calibration.json", soft_fused_info)
        extra_report_paths["hierarchical_soft_fused_element_per_class_report"] = str(
            args.out / "hierarchical_soft_fused_element_per_class_report.csv"
        )
        extra_report_paths["hierarchical_soft_fused_element_confusion_matrix"] = str(
            args.out / "hierarchical_soft_fused_element_confusion_matrix.csv"
        )
        extra_report_paths["hierarchical_soft_fused_element_calibration"] = str(
            args.out / "hierarchical_soft_fused_element_calibration.json"
        )

    save_json(
        args.out / "metrics.json",
        {
            "test_results": test_results,
            "hierarchical_fused_element": fused_info,
            "hierarchical_soft_fused_element": soft_fused_info,
            "best_val_element_accuracy": float(max(history.history.get("val_element_masked_accuracy", [0.0]))),
            "epochs_trained": int(len(history.history.get("loss", []))),
            "head_classes": head_classes,
            "schema_path": str(args.out / "framework_schema.json"),
            "element_per_class_report": str(args.out / "element_per_class_report.csv"),
            "element_confusion_matrix": str(args.out / "element_confusion_matrix.csv"),
            **extra_report_paths,
        },
    )

    print("\nTest results:")
    for key, value in sorted(test_results.items()):
        print(f"  {key}: {value:.4f}")

    # ── save split indices for provenance ──────────────────────────────────────
    np.save(args.out / "provenance_train_idx.npy", train_idx)
    np.save(args.out / "provenance_val_idx.npy",   val_idx)
    np.save(args.out / "provenance_test_idx.npy",  test_idx)
    print("Saved split indices (provenance_train/val/test_idx.npy)")

    # ── RUN_CARD.md ────────────────────────────────────────────────────────────
    best_val_acc = float(max(history.history.get("val_element_masked_accuracy", [0.0])))
    test_elem_acc = float(test_results.get("element_masked_accuracy",
                          test_results.get("element_accuracy", 0.0)))
    run_card = f"""# RUN CARD — Canonical Genome Reader Checkpoint

## Identity
- Run dir  : {args.out}
- Seed     : {args.seed}
- Date     : {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}

## Architecture
- D_MODEL={args.d_model}  N_STAGES={args.n_stages}  BLOCKS_PER_STAGE={args.blocks_per_stage}
- Sequence branch: 501 bp one-hot CNN
- KG branch: context-only node embeddings (feature_dim=76)
- Table-window branch + start-stop contrast branch
- Multitask heads: element(5), organism({len(encoders['organism'])}), kingdom({len(encoders['kingdom'])}), human_specific_element(3),
  aux_element_coarse(7), bio_element_group(4), pair_stac_stoc(2), utr_subtype(2), cds_detector(2)

## Training config
- Optimizer : AdamW  lr={args.lr}  weight_decay={args.weight_decay}
- Batch size: {args.batch_size}  Epochs: {args.epochs}  Patience: {args.patience}
- Dropout   : {args.dropout}  Label smoothing: {args.element_label_smoothing}
- Focal loss gamma: {args.element_focal_loss_gamma}
- Data cap  : max={args.max_balanced_per_organism_element_train}/min={args.min_balanced_per_organism_element_train} per (organism, element) combo
- Organism GRL λ={args.organism_gradient_reversal_lambda}  Kingdom GRL λ={args.kingdom_gradient_reversal_lambda}
- Kingdom label conditioning: {args.kingdom_label_conditioning}
- Profile normalization: {args.organism_profile_normalization}

## Primary metric — IMPORTANT
The training history column `val_element_masked_accuracy` is the HONEST main-element
accuracy: it applies the element sample_weight mask before computing accuracy, excluding
the ~70k zero-weight CDS/human-specific dummy rows that dilute Keras's built-in string
metric. Do NOT report `val_element_accuracy` from older runs as a comparable number —
it is an artifact (~0.62) not the real accuracy (~0.72).

## Final metrics (clean masked method)
- Best val_element_masked_accuracy : {best_val_acc:.4f}
- Test element_masked_accuracy     : {test_elem_acc:.4f}

## Provenance files (in this run dir)
- provenance_train_idx.npy  — balanced train indices used for training
- provenance_val_idx.npy    — balanced val indices used for checkpointing
- provenance_test_idx.npy   — balanced test indices used for final evaluation
- organism_profile_normalization.npz — per-organism z-score stats (train-split only)
- run_config.json           — full hyperparameter record
- checkpoints/best_model.keras — checkpoint saved at best val_element_masked_accuracy
"""
    (args.out / "RUN_CARD.md").write_text(run_card)
    print("Saved RUN_CARD.md")
    print(f"\nSaved model/framework outputs to {args.out}")


if __name__ == "__main__":
    main()
