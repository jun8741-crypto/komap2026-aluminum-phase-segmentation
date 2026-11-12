import numpy as np
from komap import cv


def test_same_alloy_same_fold():
    names = [f"A{a}_step{s}_sample1_image.png" for a in range(1, 31) for s in (1, 2)]
    folds = cv.make_folds(names, n_folds=5, seed=42)
    alloys = [int(n[1:].split("_")[0]) for n in names]
    for a in set(alloys):
        assert len({f for f, al in zip(folds, alloys) if al == a}) == 1


def test_folds_are_deterministic():
    names = [f"A{a}_step{(a % 4) + 1}_sample1_image.png" for a in range(1, 41)]
    assert np.array_equal(cv.make_folds(names, 5, 42), cv.make_folds(names, 5, 42))


def test_overlap_tags():
    tr = ["A1_step1_sample1_image.png", "A2_step1_sample1_image.png"]
    te = ["A2_step3_sample2_image.png"]
    tags = cv.overlap_tags(tr, te)
    assert tags[tr[0]] is False and tags[tr[1]] is True
