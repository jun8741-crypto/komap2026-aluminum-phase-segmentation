"""교차검증 분할. 그룹 = 합금 ID(같은 합금은 같은 fold), 층 = step(fold마다 step 비율 유지)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from komap import io
from komap.config import N_FOLDS, SEED


def make_folds(names, n_folds=N_FOLDS, seed=SEED):
    meta = [io.parse_name(n) for n in names]
    groups = np.array([m["alloy"] for m in meta])
    strata = np.array([m["step"] for m in meta])
    folds = np.full(len(names), -1, int)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(sgkf.split(np.zeros(len(names)), strata, groups)):
        folds[va] = k
    assert (folds >= 0).all()
    return folds


def overlap_tags(train_names, test_names):
    test_alloys = {io.parse_name(n)["alloy"] for n in test_names}
    return {n: io.parse_name(n)["alloy"] in test_alloys for n in train_names}


def build_fold_table():
    names = [p.name for p in io.list_split("train")]
    test = [p.name for p in io.list_split("test")]
    folds = make_folds(names)
    tags = overlap_tags(names, test)
    rows = [{"name": n, **io.parse_name(n), "fold": int(f), "alloy_in_test": tags[n]} for n, f in zip(names, folds)]
    return pd.DataFrame(rows)
