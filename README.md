# matilda-sc

Multi-task framework for single-cell **multimodal** data (RNA / ADT / ATAC):
joint cell-type **classification**, **dimension reduction**, **feature selection**,
data **simulation**, and **augmentation** in one model.

> The import name is `matilda`; the PyPI distribution is `matilda-sc`.
> The model is unchanged from the published engine — this package modernizes the
> call side (importability, packaging, I/O, and return objects).

## Install

During development / testing (from GitHub):

```bash
pip install "git+https://github.com/DSichang/matilda-sc.git"
```

Once published to PyPI:

```bash
pip install matilda-sc
```

## Quickstart (object API — recommended)

Work with in-memory `AnnData` (or arrays, or file paths) and get results back as objects:

```python
import matilda

# rna/adt/atac: AnnData | ndarray | scipy.sparse | path | None
# labels: a vector, an `.obs` column name, or a .csv path (string or numeric labels)
fit = matilda.train(rna, adt=adt, atac=atac, labels="cell_type")

res = matilda.task(q_rna, adt=q_adt, atac=q_atac, labels=q_labels,
                   model=fit, classification=True, query=True)

res.predictions        # DataFrame: cell_id, real, predicted, probability
res.celltype_accuracy  # DataFrame: celltype, accuracy, n
```

The modality combination is inferred automatically (RNA only → RNA-only model; +ADT →
CITE-seq; +ATAC → SHARE-seq; +both → TEA-seq). Other tasks share the same call —
`task(..., dim_reduce=True, fs=True, simulation=True, simulation_ct="B.Naive")` populates
`res.latent`, `res.markers`, and `res.simulated`. Pass `out_dir=` to also write the
artifacts to disk; otherwise the trained model lives in a temp dir (`fit.model_dir`) for
the session and is cleaned up at exit.

I/O helpers in `matilda.io` convert to/from the engine's format:
`read_matilda_h5`, `to_matilda_h5`, `to_matilda_cty`, `from_10x(dir)` (reads ADT/ATAC too).

## Lower-level path-based API

The original engine functions remain available and take file paths:

```python
from matilda import main_train, main_task
main_train("train_rna.h5", "train_adt.h5", "train_atac.h5", "train_cty.csv", seed=1)
main_task("test_rna.h5", "test_adt.h5", "test_atac.h5", "test_cty.csv",
          classification=True, query=True, seed=1)
```

RNA-only runs use `rna_train` / `rna_task` (same signatures without the ADT/ATAC arguments).
These write outputs to `../trained_model/` and `../output/` relative to the working
directory; the object API above wraps this and returns the results instead.

## Status

Work in progress: call-side modernization of the published Matilda engine. The
numerical model is frozen; results are bit-identical to the original engine for a
given device + seed + library versions.
