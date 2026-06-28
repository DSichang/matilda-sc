"""matilda.transfer: (A) with fully-matching features it equals plain train+task;
(B) with partial overlap (query misses some reference features AND adds extras) it runs on
the intersection — no padding, no dim mismatch.

    PYTHONPATH=<repo>/src MATILDA_DEMO=<demo_dir> python tests/verify_transfer.py
"""
import os

import numpy as np
import pandas as pd

from matilda import io, transfer, train, task

DATA = os.environ.get("MATILDA_DEMO", "/media/disk2/Sichang/.cache/matilda/matilda_teaseq_demo")
SEED = 1
EPOCHS = 2


def labels_of(csv):
    return pd.read_csv(csv, header=None, index_col=False).iloc[1:, 1].tolist()


ref = {m: io.read_matilda_h5(f"{DATA}/train_{m}.h5") for m in ("rna", "adt", "atac")}
qry = {m: io.read_matilda_h5(f"{DATA}/test_{m}.h5") for m in ("rna", "adt", "atac")}
trl = labels_of(f"{DATA}/train_cty.csv")
tel = labels_of(f"{DATA}/test_cty.csv")
N = ref["rna"].n_vars

# (A) full-feature transfer must equal a plain train(ref) + task(query)
res_full = transfer(ref, qry, labels=trl, query_labels=tel, epochs=EPOCHS, seed=SEED)
acc_full = float((res_full.predictions["predicted"] == res_full.predictions["real"]).mean())
print("transfer (full features): common=%s  acc=%.4f" % (res_full.common_features, acc_full))

fit = train(ref["rna"], adt=ref["adt"], atac=ref["atac"], labels=trl, epochs=EPOCHS, seed=SEED)
res_base = task(qry["rna"], adt=qry["adt"], atac=qry["atac"], labels=tel, model=fit,
                classification=True, query=True, seed=SEED)
assert np.array_equal(res_full.predictions["predicted"].values,
                      res_base.predictions["predicted"].values), "transfer != baseline on full features"
print("  -> transfer == plain train+task when features match  (acc=%.4f) OK"
      % float((res_base.predictions["predicted"] == res_base.predictions["real"]).mean()))

# (B) partial overlap: drop the first 500 RNA features from the query, rename the next 300
#     to fake names (extras not in the reference). common RNA should be N-800, and it must run.
q2 = {m: qry[m].copy() for m in qry}
keep = list(qry["rna"].var_names[500:])                 # query no longer has the first 500
q2["rna"] = qry["rna"][:, keep].copy()
vn = list(q2["rna"].var_names)
for i in range(300):
    vn[i] = "FAKE_%d" % i                               # 300 extras absent from the reference
q2["rna"].var_names = vn

res_p = transfer(ref, q2, labels=trl, query_labels=tel, epochs=EPOCHS, seed=SEED)
acc_p = float((res_p.predictions["predicted"] == res_p.predictions["real"]).mean())
print("transfer (partial RNA overlap): common=%s  acc=%.4f" % (res_p.common_features, acc_p))
exp = N - 800
assert res_p.common_features["rna"] == exp, (res_p.common_features["rna"], exp)
assert res_p.common_features["adt"] == ref["adt"].n_vars and res_p.common_features["atac"] == ref["atac"].n_vars
assert len(res_p.predictions) == q2["rna"].n_obs
print("  -> partial-overlap transfer ran; RNA common = %d (= %d - 800), ADT/ATAC full. n=%d OK"
      % (res_p.common_features["rna"], N, len(res_p.predictions)))
print("TRANSFER OK")
