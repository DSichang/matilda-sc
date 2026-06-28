"""matilda.api — object-in / results-out wrappers around the (unchanged) engine.

``train`` / ``task`` accept in-memory ``AnnData`` / arrays / paths, stage them into the
engine's bespoke ``.h5`` + cty layout via :mod:`matilda.io`, run the **unmodified**
engine (``main_train`` / ``main_task`` / ``rna_train`` / ``rna_task``) in a temporary
run directory, and return :class:`TrainResult` / :class:`TaskResult` objects built from
the engine's outputs. The numerical model is untouched; results are bit-identical to the
path-based API for the same inputs + device + seed.

Modality dispatch is automatic from which of ``adt`` / ``atac`` are provided:
RNA only -> ``rna_*``; RNA+ADT -> CITEseq; RNA+ATAC -> SHAREseq; RNA+ADT+ATAC -> TEAseq.

Thread-safety: the engine writes to CWD-relative paths, so ``train``/``task`` change the
process working directory while running and are serialized by a module lock. They are safe
to call from multiple threads (one runs at a time) but should not be parallelised for speed.
"""
from __future__ import annotations

import os
import re
import glob
import atexit
import shutil
import tempfile
import warnings
import threading
import contextlib
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from . import io
from .main_matilda_train import main_train
from .main_matilda_task import main_task
from .main_matilda_rna_train import rna_train
from .main_matilda_rna_task import rna_task

__all__ = ["train", "task", "transfer", "TrainResult", "TaskResult", "resolve_device"]

_MODE_DIRS = ("TEAseq", "CITEseq", "SHAREseq", "rna_only", "RNAseq")

# The engine resolves paths against the process CWD, which train()/task() temporarily
# change. Serialize engine runs so concurrent calls don't corrupt each other's CWD.
_ENGINE_LOCK = threading.Lock()

# train(out_dir=None) persists the model to a temp dir that must outlive the call (task()
# reads it back). Track these and remove them at process exit so they don't accumulate.
_TEMP_MODEL_DIRS: List[str] = []


@atexit.register
def _cleanup_temp_model_dirs():
    for d in _TEMP_MODEL_DIRS:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- results

@dataclass
class TrainResult:
    """Handle returned by :func:`train`."""

    model_path: str
    model_dir: str
    mode: str
    classes: List[str]
    out_dir: Optional[str] = None
    train_acc: Optional[float] = None  # not populated under approach (A)

    def __repr__(self):
        return ("TrainResult(mode=%r, n_classes=%d, model_path=%r)"
                % (self.mode, len(self.classes), self.model_path))


@dataclass
class TaskResult:
    """Handle returned by :func:`task`. Only requested tasks are populated."""

    mode: str
    predictions: Optional[pd.DataFrame] = None
    celltype_accuracy: Optional[pd.DataFrame] = None
    latent: Optional[pd.DataFrame] = None
    latent_labels: Optional[pd.DataFrame] = None
    markers: Optional[pd.DataFrame] = None
    simulated: Optional[Dict[str, pd.DataFrame]] = None
    out_dir: Optional[str] = None
    common_features: Optional[Dict[str, int]] = None  # set by transfer(): per-modality #common

    def __repr__(self):
        got = [n for n in ("predictions", "celltype_accuracy", "latent", "markers",
                           "simulated") if getattr(self, n) is not None]
        return "TaskResult(mode=%r, populated=%s)" % (self.mode, got)


# --------------------------------------------------------------------------- helpers

def resolve_device(device="auto"):
    """Validate/normalize ``device`` and return ``"cuda"`` or ``"cpu"``.

    ``"auto"`` uses CUDA if available. ``"cuda"`` errors if CUDA is absent. ``"cpu"`` is
    best-effort: the engine binds tensor types to CUDA at *import* (``util.py``), so a
    guaranteed CPU run needs ``CUDA_VISIBLE_DEVICES=""`` set *before* ``import matilda``.
    """
    import torch

    d = (device or "auto").lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available.")
        return "cuda"
    if d == "cpu":
        # CUDA was bound at import; warn whenever it is visible (the env var must have
        # been cleared BEFORE import to actually take effect).
        if torch.cuda.is_available():
            warnings.warn(
                "device='cpu' is best-effort: matilda's engine selects the device when it "
                "is first imported. For a guaranteed CPU run, set CUDA_VISIBLE_DEVICES='' "
                "BEFORE `import matilda`; otherwise the GPU may still be used.",
                RuntimeWarning,
            )
        return "cpu"
    raise ValueError("device must be 'auto', 'cpu' or 'cuda', got %r" % (device,))


def _abs(x):
    """Absolutize a string path input; pass non-strings (AnnData/array/None) through."""
    return os.path.abspath(x) if isinstance(x, str) else x


def _mode(adt, atac):
    if adt is not None and atac is not None:
        return "TEAseq"
    if adt is not None:
        return "CITEseq"
    if atac is not None:
        return "SHAREseq"
    return "rna_only"


def _resolve_labels(labels, rna):
    """Return a list of per-cell label strings, or ``None``. Raise on missing values."""
    if labels is None:
        return None
    if isinstance(labels, str):
        obs = getattr(rna, "obs", None)
        if obs is not None and labels in obs.columns:
            vec = list(obs[labels])
        elif os.path.isfile(labels) and labels.endswith(".csv"):
            raw = pd.read_csv(labels, header=None, index_col=False)
            vec = list(raw.iloc[1:, 1])
        else:
            raise ValueError(
                "labels=%r is neither an .obs column of the AnnData nor a path to a .csv" % labels
            )
    else:
        vec = list(labels)
    if len(vec) and pd.isna(pd.array(vec, dtype="object")).any():
        raise ValueError(
            "labels contain missing values (NaN/None). Drop or annotate those cells before "
            "calling train()/task() — Matilda has no 'unlabelled' class."
        )
    return vec


def _classes_from_labels(labels_vec):
    """Class order as the engine binds it: alphabetical categories of the STRING labels."""
    return list(pd.Categorical([str(x) for x in labels_vec]).categories)


def _infer_n_cells(rna, labels_vec):
    if labels_vec is not None:
        return len(labels_vec)
    if hasattr(rna, "n_obs"):
        return int(rna.n_obs)
    return None


def _stage_modality(obj, name, stage_dir, n_cells):
    """Materialise a modality to a Matilda ``.h5`` and return its absolute path (or 'NULL')."""
    if obj is None:
        return "NULL"
    if isinstance(obj, str):                       # already absolutized by the caller
        if os.path.isdir(obj):
            obj = io.from_10x(obj)
        elif obj.endswith(".h5ad"):
            import anndata as ad
            obj = ad.read_h5ad(obj)
        else:
            return obj                              # assume it is already a Matilda .h5
    path = os.path.join(stage_dir, name + ".h5")
    return io.to_matilda_h5(obj, path, n_cells=n_cells)


def _stage_labels(labels_vec, stage_dir):
    if labels_vec is None:
        return "NULL"
    return io.to_matilda_cty(labels_vec, os.path.join(stage_dir, "cty.csv"))


@contextlib.contextmanager
def _rundir():
    """A temp tree whose ``run/`` CWD makes the engine write to ``<root>/{trained_model,output}``.

    Serialized by ``_ENGINE_LOCK`` because it mutates the process-global CWD.
    """
    with _ENGINE_LOCK:
        root = tempfile.mkdtemp(prefix="matilda_run_")
        run = os.path.join(root, "run")
        os.makedirs(run)
        for m in _MODE_DIRS:                       # pre-create so rna_train's TEAseq save can't crash
            os.makedirs(os.path.join(root, "trained_model", m), exist_ok=True)
        os.makedirs(os.path.join(root, "output"), exist_ok=True)
        try:
            prev = os.getcwd()
        except OSError:                            # original CWD was removed by another process
            prev = root
        os.chdir(run)
        try:
            yield root, run
        finally:
            try:
                os.chdir(prev)
            except OSError:
                os.chdir(tempfile.gettempdir())
            shutil.rmtree(root, ignore_errors=True)


def _write_classes(classes, where):
    pd.DataFrame(list(classes), columns=["CellType"]).to_csv(
        os.path.join(where, "real_cty.csv"), index=False, header=False
    )


# ------------------------------------------------------------------- output readers

def _read_predictions(out_root, mode, split, has_real):
    p = os.path.join(out_root, "classification", mode, split, "accuracy_each_cell.txt")
    if not os.path.isfile(p):
        raise FileNotFoundError("expected classification output not found at %s" % p)
    rows = []
    with open(p) as fh:
        for ln in fh:
            pred = re.search(r"predicted cell type:\s*(.+?)\s+probability:", ln)
            if pred is None:
                continue
            cid = re.search(r"cell ID:\s*(\d+)", ln)
            prob = re.search(r"probability:\s*([0-9.eE+-]+)", ln)
            real = re.search(r"real cell type:\s*(.+?)\s+predicted cell type:", ln)
            rows.append({
                "cell_id": int(cid.group(1)) if cid else len(rows),
                "real": real.group(1).strip() if (real and has_real) else None,
                "predicted": pred.group(1).strip(),
                "probability": float(prob.group(1)) if prob else float("nan"),
            })
    if not rows:
        raise RuntimeError("parsed 0 predictions from %s (engine wrote no rows?)" % p)
    df = pd.DataFrame(rows, columns=["cell_id", "real", "predicted", "probability"])
    # The engine emits 'real cell type: -1' for every cell when the task set collapses to a
    # single class (max code 0). Treat that sentinel as "no ground truth".
    if has_real and (df["real"].astype("string") == "-1").all():
        df["real"] = None
    return df


def _celltype_accuracy(pred_df):
    if pred_df is None or pred_df["real"].isna().all():
        return None
    rows = []
    for ct, sub in pred_df.groupby("real"):
        rows.append({"celltype": ct,
                     "accuracy": float((sub["predicted"] == ct).mean()),
                     "n": int(len(sub))})
    return pd.DataFrame(rows, columns=["celltype", "accuracy", "n"])


def _read_latent(out_root, mode, split):
    base = os.path.join(out_root, "dim_reduce", mode, split)
    latent = pd.read_csv(os.path.join(base, "latent_space.csv"), index_col=0)
    labels = None
    lp = os.path.join(base, "latent_space_label.csv")
    if os.path.isfile(lp):
        labels = pd.read_csv(lp, index_col=0)
    return latent, labels


def _read_markers(out_root, mode, split):
    base = os.path.join(out_root, "marker", mode, split)
    rows = []
    for f in sorted(glob.glob(os.path.join(base, "fs.celltype_*.csv"))):
        ct = os.path.basename(f)[len("fs.celltype_"):-len(".csv")]
        d = pd.read_csv(f, index_col=0)
        col = d.columns[0]
        for feat, val in zip(d.index, d[col]):
            rows.append({"celltype": ct, "feature": feat, "importance": float(val)})
    return pd.DataFrame(rows, columns=["celltype", "feature", "importance"]) if rows else None


def _read_simulation(out_root, mode, split, include_real=False):
    base = os.path.join(out_root, "simulation_result", mode, split)
    out = {}
    for m in ("rna", "adt", "atac"):
        f = os.path.join(base, "sim_data_%s.csv" % m)
        if os.path.isfile(f):
            out[m] = pd.read_csv(f, index_col=0)
    lf = os.path.join(base, "sim_label.csv")
    if os.path.isfile(lf):
        out["label"] = pd.read_csv(lf, index_col=0)
    if include_real:                                 # the real reference cells, same feature space
        for m in ("rna", "adt", "atac"):
            f = os.path.join(base, "real_data_%s.csv" % m)
            if os.path.isfile(f):
                out["real_%s" % m] = pd.read_csv(f, index_col=0)
        rlf = os.path.join(base, "real_label.csv")
        if os.path.isfile(rlf):
            out["real_label"] = pd.read_csv(rlf, index_col=0)
    return out or None


# --------------------------------------------------------------------------- public API

def train(rna, adt=None, atac=None, labels=None, *, batch_size=64, epochs=30, lr=0.02,
          z_dim=100, hidden_rna=185, hidden_adt=30, hidden_atac=185, seed=1,
          augmentation=True, out_dir=None, device="auto"):
    """Train Matilda from in-memory objects and return a :class:`TrainResult`.

    ``rna`` (required) and optional ``adt`` / ``atac`` each accept
    ``AnnData | ndarray | scipy.sparse | path | None``. ``labels`` (required) accepts a
    vector, an ``.obs`` column name (resolved against ``rna``), or a ``.csv`` path; labels
    may be strings or numbers but must not contain missing values.

    The trained model is persisted to ``out_dir`` if given (relative paths are resolved
    against the caller's working directory), else to a temporary directory whose path is in
    ``result.model_dir`` and which is removed at process exit — so consume it within the
    same session or pass ``out_dir`` to keep it.

    ``device``: ``"auto"`` (GPU if available), ``"cuda"``, or ``"cpu"`` (best-effort — see
    :func:`resolve_device`).
    """
    if labels is None:
        raise ValueError("train() requires labels.")
    resolve_device(device)
    rna, adt, atac = _abs(rna), _abs(adt), _abs(atac)
    out_dir = os.path.abspath(out_dir) if out_dir else None
    mode = _mode(adt, atac)
    labels_vec = _resolve_labels(labels, rna)
    n_cells = _infer_n_cells(rna, labels_vec)

    with _rundir() as (root, run):
        stage = os.path.join(root, "stage")
        os.makedirs(stage, exist_ok=True)
        rna_p = _stage_modality(rna, "rna", stage, n_cells)
        adt_p = _stage_modality(adt, "adt", stage, n_cells)
        atac_p = _stage_modality(atac, "atac", stage, n_cells)
        cty_p = _stage_labels(labels_vec, stage)

        if mode == "rna_only":
            rna_train(rna_p, cty_p, batch_size=batch_size, epochs=epochs, lr=lr,
                      z_dim=z_dim, hidden_rna=hidden_rna, seed=seed, augmentation=augmentation)
        else:
            main_train(rna_p, adt_p, atac_p, cty_p, batch_size=batch_size, epochs=epochs,
                       lr=lr, z_dim=z_dim, hidden_rna=hidden_rna, hidden_adt=hidden_adt,
                       hidden_atac=hidden_atac, seed=seed, augmentation=augmentation)

        # Class order = how the engine binds it: alphabetical categories of the STRING
        # labels (main_train writes this to real_cty.csv; rna_train does not, so derive it).
        classes = _classes_from_labels(labels_vec)
        src = os.path.join(root, "trained_model", mode, "model_best.pth.tar")
        if not os.path.isfile(src):
            raise RuntimeError("training did not produce a model at %s" % src)

        if out_dir:
            dest_root = out_dir
        else:
            dest_root = tempfile.mkdtemp(prefix="matilda_model_")
            _TEMP_MODEL_DIRS.append(dest_root)
        model_dir = os.path.join(dest_root, "trained_model", mode)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model_best.pth.tar")
        shutil.copy2(src, model_path)
        _write_classes(classes, dest_root)     # keep classes alongside the model

    return TrainResult(model_path=model_path, model_dir=model_dir, mode=mode,
                       classes=classes, out_dir=out_dir)


def task(rna, adt=None, atac=None, labels=None, *, model=None, classification=False,
         query=False, fs=False, fs_method="IntegratedGradient", dim_reduce=False,
         simulation=False, simulation_ct=None, simulation_num=100, include_real=False,
         batch_size=64, z_dim=100, hidden_rna=185, hidden_adt=30, hidden_atac=185,
         seed=1, out_dir=None, device="auto"):
    """Run one or more tasks with a trained model and return a :class:`TaskResult`.

    ``model`` is a :class:`TrainResult` from :func:`train` (or a path to a model dir
    containing ``trained_model/<mode>/model_best.pth.tar`` + ``real_cty.csv``). The task
    flags are independent — several may be ``True`` in one call. ``query`` marks the input
    as a held-out query set (vs reference). Outputs are copied to ``out_dir`` if given
    (relative paths resolved against the caller's working directory).

    ``fs_method``: ``"IntegratedGradient"`` (default) or ``"Saliency"``.
    ``simulation``: requires ``labels`` and ``simulation_ct`` — a cell type present in the
    model's classes *and* in ``labels``, or the sentinel ``"-1"`` to simulate all cells.
    ``include_real=True`` also returns the real reference cells (``result.simulated`` then
    holds ``real_rna`` / ``real_adt`` / ``real_atac`` / ``real_label`` alongside the
    simulated ones), in the same feature space — useful for real-vs-simulated comparison.
    ``device``: see :func:`resolve_device`.
    """
    if model is None:
        raise ValueError("task() requires model= (a TrainResult from train()).")
    resolve_device(device)
    rna, adt, atac = _abs(rna), _abs(adt), _abs(atac)
    out_dir = os.path.abspath(out_dir) if out_dir else None
    mode = _mode(adt, atac)

    if isinstance(model, TrainResult):
        if model.mode != mode:
            raise ValueError("model was trained for mode %r but the task modalities imply %r"
                             % (model.mode, mode))
        src_model_dir = model.model_dir
        classes = list(model.classes)
    else:
        model = os.path.abspath(str(model))
        src_model_dir = os.path.join(model, "trained_model", mode)
        cty_file = os.path.join(model, "real_cty.csv")
        classes = pd.read_csv(cty_file, header=None)[0].tolist() if os.path.isfile(cty_file) else None

    labels_vec = _resolve_labels(labels, rna)
    n_cells = _infer_n_cells(rna, labels_vec)
    split = "query" if query else "reference"

    if simulation:
        if simulation_ct is None:
            raise ValueError(
                "simulation=True requires simulation_ct: a cell type in the model's classes "
                "and in labels, or '-1' to simulate all cells."
            )
        if simulation_ct != "-1":
            if labels_vec is None:
                raise ValueError("simulation=True requires labels= so the target cell type "
                                 "can be located in the task data.")
            if classes is not None and str(simulation_ct) not in [str(c) for c in classes]:
                raise ValueError("simulation_ct=%r is not in the model classes %r "
                                 "(use '-1' to simulate all cells)." % (simulation_ct, classes))
            if str(simulation_ct) not in [str(v) for v in labels_vec]:
                raise ValueError("simulation_ct=%r has no cells in the task labels."
                                 % (simulation_ct,))

    with _rundir() as (root, run):
        # place the trained model + class list where the engine looks for them
        dst_model_dir = os.path.join(root, "trained_model", mode)
        os.makedirs(dst_model_dir, exist_ok=True)
        src_ckpt = os.path.join(src_model_dir, "model_best.pth.tar")
        if not os.path.isfile(src_ckpt):
            raise FileNotFoundError("no trained model at %s; train first (and check the "
                                    "modalities match the trained mode)." % src_ckpt)
        shutil.copy2(src_ckpt, os.path.join(dst_model_dir, "model_best.pth.tar"))
        if classes is None:
            raise ValueError("could not determine the class list; pass a TrainResult as model=.")
        _write_classes(classes, run)

        stage = os.path.join(root, "stage")
        os.makedirs(stage, exist_ok=True)
        rna_p = _stage_modality(rna, "rna", stage, n_cells)
        adt_p = _stage_modality(adt, "adt", stage, n_cells)
        atac_p = _stage_modality(atac, "atac", stage, n_cells)
        cty_p = _stage_labels(labels_vec, stage)

        if mode == "rna_only":
            rna_task(rna_p, cty_p, batch_size=batch_size, z_dim=z_dim, hidden_rna=hidden_rna,
                     seed=seed, classification=classification, query=query, fs=fs,
                     fs_method=fs_method, dim_reduce=dim_reduce, simulation=simulation,
                     simulation_ct=simulation_ct, simulation_num=simulation_num)
        else:
            main_task(rna_p, adt_p, atac_p, cty_p, batch_size=batch_size, z_dim=z_dim,
                      hidden_rna=hidden_rna, hidden_adt=hidden_adt, hidden_atac=hidden_atac,
                      seed=seed, classification=classification, query=query, fs=fs,
                      fs_method=fs_method, dim_reduce=dim_reduce, simulation=simulation,
                      simulation_ct=simulation_ct, simulation_num=simulation_num)

        out_root = os.path.join(root, "output")
        res = TaskResult(mode=mode)
        if classification:
            res.predictions = _read_predictions(out_root, mode, split, has_real=(cty_p != "NULL"))
            res.celltype_accuracy = _celltype_accuracy(res.predictions)
        if dim_reduce:
            res.latent, res.latent_labels = _read_latent(out_root, mode, split)
        if fs:
            res.markers = _read_markers(out_root, mode, split)
        if simulation:
            res.simulated = _read_simulation(out_root, mode, split, include_real=include_real)

        if out_dir:
            dst = os.path.join(out_dir, "output")
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(out_root, dst)
            res.out_dir = out_dir

    return res


def _intersect_anndata(ref, qry):
    """Restrict two AnnData to their common features (reference order); return (ref_sub, qry_sub)."""
    if not hasattr(ref, "var_names") or not hasattr(qry, "var_names"):
        raise ValueError("transfer() needs AnnData inputs with feature names (var_names).")
    qset = set(str(v) for v in qry.var_names)
    seen, common = set(), []
    for v in (str(x) for x in ref.var_names):
        if v in qset and v not in seen:
            common.append(v); seen.add(v)
    if not common:
        raise ValueError("reference and query share no common features in a modality.")
    return ref[:, common].copy(), qry[:, common].copy()


def transfer(reference, query, labels=None, query_labels=None, *,
             classification=True, dim_reduce=False, fs=False, simulation=False,
             fs_method="IntegratedGradient", simulation_ct=None, simulation_num=100,
             include_real=False, batch_size=64, epochs=30, lr=0.02, z_dim=100,
             hidden_rna=185, hidden_adt=30, hidden_atac=185, seed=1,
             out_dir=None, device="auto"):
    """Transfer labels (and/or other tasks) from a labeled reference to a query whose
    features only partially overlap.

    Computes the per-modality **feature intersection** (in reference order), trains a model
    on the intersection, and applies it to the query — real values only, **no zero-padding**.
    This is the right approach when the query both misses some reference features *and* adds
    others. Because the model is trained on the reference∩query feature set, both ``reference``
    and ``query`` are needed together (a different query → a different intersection → its own
    model).

    Parameters
    ----------
    reference, query : AnnData | dict
        Each is an ``AnnData`` (RNA only) or a dict ``{"rna": AnnData, "adt": AnnData,
        "atac": AnnData}`` (ADT/ATAC optional). Only modalities present in **both** are used.
    labels : reference cell-type labels (vector / ``.obs`` column name / ``.csv`` path) — required.
    query_labels : optional ground-truth labels for the query (adds the accuracy report).

    Returns
    -------
    :class:`TaskResult` for the query, with ``.common_features`` recording how many features
    each modality kept after intersection.
    """
    if labels is None:
        raise ValueError("transfer() requires labels (the reference's cell-type labels).")
    ref = reference if isinstance(reference, dict) else {"rna": reference}
    qry = query if isinstance(query, dict) else {"rna": query}
    if ref.get("rna") is None or qry.get("rna") is None:
        raise ValueError("reference and query must each include an 'rna' modality.")
    for m in ("adt", "atac"):
        if (ref.get(m) is not None) != (qry.get(m) is not None):
            warnings.warn("modality %r is present in only one of reference/query; it will be "
                          "dropped (only modalities present in both are used)." % m, RuntimeWarning)
    mods = [m for m in ("rna", "adt", "atac") if ref.get(m) is not None and qry.get(m) is not None]

    ref_sub, qry_sub, common = {}, {}, {}
    for m in mods:
        rs, qs = _intersect_anndata(ref[m], qry[m])
        ref_sub[m], qry_sub[m] = rs, qs
        common[m] = int(rs.n_vars)

    fit = train(ref_sub["rna"], adt=ref_sub.get("adt"), atac=ref_sub.get("atac"),
                labels=labels, batch_size=batch_size, epochs=epochs, lr=lr, z_dim=z_dim,
                hidden_rna=hidden_rna, hidden_adt=hidden_adt, hidden_atac=hidden_atac,
                seed=seed, device=device)
    res = task(qry_sub["rna"], adt=qry_sub.get("adt"), atac=qry_sub.get("atac"),
               labels=query_labels, model=fit, classification=classification, query=True,
               fs=fs, fs_method=fs_method, dim_reduce=dim_reduce, simulation=simulation,
               simulation_ct=simulation_ct, simulation_num=simulation_num,
               include_real=include_real, batch_size=batch_size, z_dim=z_dim,
               hidden_rna=hidden_rna, hidden_adt=hidden_adt, hidden_atac=hidden_atac,
               seed=seed, out_dir=out_dir, device=device)
    res.common_features = common
    return res
