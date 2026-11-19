"""Step 6 analysis functions on synthetic tables where the right answer is known."""
import numpy as np
import pandas as pd

from metabric.agreement import analysis as A


def _preds(n=800, hard_effect=True, seed=0):
    """Two repeats x one split each; 4 members. Half the patients are 'hard': members disagree on
    them and (if hard_effect) are also wrong more often."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-1.5 * z))).astype(int)
    hard = rng.random(n) < 0.5
    rows = []
    for r in (0, 1):
        for m in range(4):
            noise = rng.normal(size=n) * np.where(hard, 1.5, 0.2)
            signal = np.where(hard & hard_effect, 0.2 * z, 1.5 * z) if hard_effect else 1.5 * z
            p = 1 / (1 + np.exp(-(signal + noise)))
            rows.append(pd.DataFrame({"candidate": f"C|m{m}", "split_id": f"r{r}o0", "repeat": r, "patient_id": np.arange(n),
                                      "y": y, "p": p}))
    return pd.concat(rows, ignore_index=True)


def test_disagreement_flag_detects_error_link_and_not_noise():
    procs = [f"C|m{m}" for m in range(4)]
    q, v = A.disagreement(_preds(hard_effect=True), procs, 200, np.random.default_rng(1))
    assert v["flag_earns_place"] and v["spearman_sd_vs_excess_error"] > 0
    assert q.excess.iloc[-1] > q.excess.iloc[0]                      # most-disagreeing quintile has more excess error
    # disagreement that is pure noise (signal intact for everyone) should not earn a flag on excess error
    _, v2 = A.disagreement(_preds(hard_effect=False, seed=3), procs, 200, np.random.default_rng(1))
    assert v2["spearman_sd_vs_excess_error"] < v["spearman_sd_vs_excess_error"]


def test_robustness_mean_of_members_is_more_stable():
    pr = _preds(hard_effect=False)
    procs = [f"C|m{m}" for m in range(4)]
    r = A.robustness(pr, procs, {"mean of members": procs}).set_index("model")
    assert r.loc["mean of members", "mean_abs_change"] < r.loc[procs, "mean_abs_change"].min()


def test_importance_matrix_overlap_and_ablation():
    summ = pd.DataFrame({"procedure": ["C|a"] * 4 + ["C|b"] * 4, "item": list("wxyz") * 2,
                         "mean": [0.3, 0.2, 0.1, 0.0, 0.3, 0.0, 0.2, 0.1]})
    mat, cons = A.importance_matrix(summ, ["C|a", "C|b"], 2)
    assert list(cons.item[:1]) == ["w"] and cons.set_index("item").loc["w", "share_in_top"] == 1.0
    ov = A.rank_overlap(summ, ["C|a", "C|b"], ks=(2,))
    assert ov.jaccard_top2.iloc[0] == 1 / 3                          # {w,x} vs {w,y}
    auc = pd.DataFrame({"A": [0.70, 0.71, 0.72, 0.70], "B": [0.75, 0.76, 0.77, 0.75]}, index=list("abcd"))
    ab = A.ablation(auc, [("A", None), ("B", "A")], 100, 25).set_index("procedure")
    assert abs(ab.loc["B", "diff"] - 0.05) < 1e-9 and ab.loc["B", "p"] < 0.05 and ab.loc["A", "vs"] == ""
