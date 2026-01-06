import sys
import math
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


# ============================================================
# Versioning / metadata
# ============================================================

def get_versions() -> dict:
    return {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def print_versions() -> dict:
    info = get_versions()
    print("Python:", info["python"])
    print("numpy:", info["numpy"])
    print("pandas:", info["pandas"])
    print("scikit-learn:", info["sklearn"])
    return info


# ============================================================
# Data
# ============================================================

def load_wdbc(path: str):
    """Load WDBC in UCI format: [id, label, 30 features]. Label mapping: M=1, B=0."""
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, 2:].values.astype(float)
    y = (df.iloc[:, 1].values == "M").astype(int)  # 1=M (malignant), 0=B (benign)
    return X, y, df


def stratified_split_indices(
    y: np.ndarray, seed: int, train_frac: float = 0.6, cal_frac: float = 0.2
):
    """Stratified train/cal/test split with proportions train_frac/cal_frac/(rest)."""
    idx = np.arange(len(y))

    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, temp_idx = next(sss1.split(idx, y))

    y_temp = y[temp_idx]
    cal_prop_within_temp = cal_frac / (1.0 - train_frac)

    sss2 = StratifiedShuffleSplit(
        n_splits=1, train_size=cal_prop_within_temp, random_state=seed + 10_000
    )
    cal_rel, test_rel = next(sss2.split(np.arange(len(temp_idx)), y_temp))

    cal_idx = temp_idx[cal_rel]
    test_idx = temp_idx[test_rel]
    return train_idx, cal_idx, test_idx


# ============================================================
# Conformal quantile
# ============================================================

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Conservative split-conformal quantile:
      k = ceil((m+1)(1-alpha)), qhat = k-th smallest calibration score.
    No randomized tie-breaking: slightly conservative, but preserves finite-sample guarantee.
    """
    scores = np.asarray(scores, dtype=float)
    m = len(scores)
    if m <= 0:
        # Should not happen for stratified WDBC; safe fallback.
        return 1.0
    k = int(math.ceil((m + 1) * (1 - alpha)))
    s_sorted = np.sort(scores)
    return float(s_sorted[min(k - 1, m - 1)])


# ============================================================
# Atomic writers
# ============================================================

def atomic_write_csv(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite to replace it.")
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_text(text: str, path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite to replace it.")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ============================================================
# Models
# ============================================================

def fit_model_and_predict_proba(model_name: str, Xtr, ytr, Xcal, Xte):
    """Fit on train only; return predict_proba on calibration and test."""
    if model_name == "logreg":
        scaler = StandardScaler().fit(Xtr)
        Xtr2 = scaler.transform(Xtr)
        Xcal2 = scaler.transform(Xcal)
        Xte2 = scaler.transform(Xte)

        model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=0)
        model.fit(Xtr2, ytr)
        return model.predict_proba(Xcal2), model.predict_proba(Xte2)

    if model_name == "hgbdt":
        model = HistGradientBoostingClassifier(
            random_state=0, max_depth=None, learning_rate=0.1, max_iter=300
        )
        model.fit(Xtr, ytr)
        return model.predict_proba(Xcal), model.predict_proba(Xte)

    raise ValueError(f"Unknown model_name: {model_name}")


def sanity_check_proba(probs: np.ndarray, name: str) -> None:
    assert probs.ndim == 2 and probs.shape[1] == 2, f"{name}: expected shape (n,2)"
    assert np.all(np.isfinite(probs)), f"{name}: contains non-finite values"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6), f"{name}: rows do not sum to 1"


# ============================================================
# SplitCP + Mondrian
# ============================================================

def splitcp_cal_scores(probs_cal: np.ndarray, ycal: np.ndarray) -> np.ndarray:
    """Nonconformity s(x,y)=1-p(y|x) on calibration."""
    p_true = probs_cal[np.arange(len(ycal)), ycal]
    return 1.0 - p_true


def predict_sets_threshold(probs_te: np.ndarray, qhat: float) -> list[list[int]]:
    """Gamma(x) = {y: 1-p_y <= qhat}  <=>  p_y >= 1-qhat. May be empty."""
    thr = 1.0 - float(qhat)
    out: list[list[int]] = []
    for p in probs_te:
        ys = [int(i) for i, pi in enumerate(p) if float(pi) >= thr]
        out.append(ys)
    return out


def mondrian_qhats(cal_scores: np.ndarray, ycal: np.ndarray, alpha: float) -> dict:
    """Per-class conformal quantiles qhat(y) using scores from that class only."""
    qh = {}
    for c in (0, 1):
        sc = cal_scores[ycal == c]
        qh[c] = conformal_quantile(sc, alpha=float(alpha))
    return qh


def mondrian_predict_sets(probs_te: np.ndarray, qhats: dict) -> list[list[int]]:
    """
    Mondrian sets: include y if 1-p_y <= qhat(y)  <=> p_y >= 1-qhat(y)
    """
    thr0 = 1.0 - float(qhats[0])
    thr1 = 1.0 - float(qhats[1])
    out: list[list[int]] = []
    for p in probs_te:
        ys = []
        if float(p[0]) >= thr0:
            ys.append(0)
        if float(p[1]) >= thr1:
            ys.append(1)
        out.append(ys)
    return out


# ============================================================
# APS (smoothed / randomized)  + two reporting modes
# ============================================================

def aps_scores_all_labels_sorted(p: np.ndarray, u: float) -> np.ndarray:
    """
    APS score for ALL labels for one sample:
      order labels by decreasing p
      score(sorted_j) = sum_{k<j} p_sorted[k] + u * p_sorted[j]
    Stable sort ensures deterministic tie behaviour.
    """
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    cumsum_prev = np.concatenate([[0.0], np.cumsum(p_sorted)[:-1]])
    scores_sorted = cumsum_prev + float(u) * p_sorted
    scores = np.empty_like(scores_sorted)
    scores[order] = scores_sorted
    return scores


def aps_cal_scores_smoothed(
    probs_cal: np.ndarray, ycal: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Smoothed APS calibration score for the true label (one u per sample)."""
    m, _K = probs_cal.shape
    scores = np.empty(m, dtype=float)
    for i in range(m):
        u = float(rng.random())
        scores_all = aps_scores_all_labels_sorted(probs_cal[i], u=u)
        scores[i] = float(scores_all[int(ycal[i])])
    return scores


def aps_predict_raw_sets_smoothed(
    probs_te: np.ndarray,
    qhat: float,
    rng: np.random.Generator,
) -> tuple[list[list[int]], np.ndarray]:
    """
    Raw APS prediction sets using smoothed scores (may be empty).
    Returns (sets_raw, raw_empty_mask).
    """
    n, K = probs_te.shape
    out: list[list[int]] = []
    raw_empty_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        p = probs_te[i]
        u = float(rng.random())
        scores_y = aps_scores_all_labels_sorted(p, u=u)

        ys_raw = [int(y) for y in range(K) if float(scores_y[y]) <= float(qhat)]
        raw_empty_mask[i] = (len(ys_raw) == 0)
        out.append(ys_raw)

    return out, raw_empty_mask


def aps_apply_argmax_fallback(
    sets_raw: list[list[int]], probs_te: np.ndarray
) -> list[list[int]]:
    """Post-processing: if empty -> {argmax}. This is NOT the same object as raw conformal sets."""
    out = []
    for s, p in zip(sets_raw, probs_te):
        if len(s) == 0:
            out.append([int(np.argmax(p))])
        else:
            out.append(s)
    return out


# ============================================================
# Metrics
# ============================================================

def set_metrics(sets: list[list[int]], y_true: np.ndarray) -> dict:
    sizes = np.array([len(s) for s in sets], dtype=int)
    contains = np.array([int(yt in s) for yt, s in zip(y_true, sets)], dtype=float)

    out = {
        "coverage": float(contains.mean()),
        "avg_set_size": float(sizes.mean()),
        "empty_rate": float((sizes == 0).mean()),   # post-processing empty rate
        "singleton_rate": float((sizes == 1).mean()),
        "ambiguous_rate": float((sizes == 2).mean()),  # binary
    }
    for c in (0, 1):
        mask = (y_true == c)
        out[f"cov_{c}"] = float(contains[mask].mean()) if mask.any() else float("nan")
    return out


# ============================================================
# Plotting
# ============================================================

def plot_with_errorbars(
    summ: pd.DataFrame,
    out: Path,
    y_col: str,
    ylabel: str,
    title: str,
    filename: str,
    add_target_line: bool = False,
) -> None:
    y_mean = f"{y_col}_mean"
    y_sd = f"{y_col}_sd"
    if y_mean not in summ.columns or y_sd not in summ.columns:
        print(f"Skip plot {filename}: missing columns {y_mean}/{y_sd}")
        return

    plt.figure()
    for (model, method), g in summ.groupby(["model", "method"], sort=True):
        g = g.sort_values("alpha").dropna(subset=[y_mean])
        if g.empty:
            continue
        label = f"{model}-{method}"
        plt.errorbar(g["alpha"], g[y_mean], yerr=g[y_sd], marker="o", capsize=3, label=label)

    if add_target_line:
        al = np.sort(summ["alpha"].unique())
        plt.plot(al, 1.0 - al, linestyle="--", label="target 1-α")

    plt.xlabel("alpha (miscoverage)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / filename, dpi=200)
    plt.close()
    print("Saved plot:", filename)


# ============================================================
# RNG scheme (deterministic)
# ============================================================

def _make_rng_base(seed: int, model_name: str, method: str) -> int:
    model_offset = 0 if model_name == "logreg" else 7
    method_offset = {
        "splitcp": 0,
        "mondrian": 3,
        "aps": 11,
    }.get(method, 99)
    return seed * 1_000_003 + model_offset + method_offset


def _alpha_key(a: float) -> int:
    return int(round(float(a) * 1_000_000))


# ============================================================
# Tables
# ============================================================

def _fmt_pm_math(mean: float, sd: float, nd: int = 3) -> str:
    if not np.isfinite(mean) or not np.isfinite(sd):
        return r"$\mathrm{NA}$"
    return rf"${mean:.{nd}f}\pm{sd:.{nd}f}$"


def write_wdbc_table_key_tex(
    summ: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Key table for main text (used by LaTeX):
      alpha | Model | Method | Emp cov | Emp cov(M) | Emp cov(B) | Avg size | Diagnostic | AUC

    Diagnostic means:
      - SplitCP / Mondrian / APS(allow-empty): empty_rate
      - APS(+fallback): aps_fallback_rate  (== raw empty rate before fallback)
    """
    alpha_order = sorted(summ["alpha"].unique())
    model_order = ["logreg", "hgbdt"]
    method_order = ["splitcp", "mondrian", "aps_allowempty", "aps_fallback"]

    def pretty_model(m: str) -> str:
        return {"logreg": "LR", "hgbdt": "HGB"}.get(m, m)

    def pretty_method(meth: str) -> str:
        return {
            "splitcp": "SplitCP",
            "mondrian": "Mondrian",
            "aps_allowempty": "APS (allow-empty)",
            "aps_fallback": "APS (+fallback)",
        }.get(meth, meth)

    lines = []
    lines.append(r"% wdbc_table_2x2_key.tex  (auto-generated; one row per alpha/model/method)")
    lines.append(r"\begin{tabular}{@{}c l l c c c c c c@{}}")
    lines.append(r"\toprule")
    lines.append(r"$\alpha$ & Model & Method & Emp.\ cov. & Emp.\ cov.(M) & Emp.\ cov.(B) & Avg.\ size & Diagnostic & AUC \\")
    lines.append(r"\midrule")

    for a in alpha_order:
        for model in model_order:
            for method in method_order:
                r = summ[(summ["alpha"] == a) & (summ["model"] == model) & (summ["method"] == method)]
                if r.empty:
                    cov = cov1 = cov0 = size = diag = auc = r"$\mathrm{NA}$"
                else:
                    r0 = r.iloc[0]
                    cov  = _fmt_pm_math(float(r0["coverage_mean"]), float(r0["coverage_sd"]), nd=3)
                    cov1 = _fmt_pm_math(float(r0["cov_1_mean"]),    float(r0["cov_1_sd"]),    nd=3)
                    cov0 = _fmt_pm_math(float(r0["cov_0_mean"]),    float(r0["cov_0_sd"]),    nd=3)
                    size = _fmt_pm_math(float(r0["avg_set_size_mean"]), float(r0["avg_set_size_sd"]), nd=3)
                    auc  = _fmt_pm_math(float(r0["auc_mean"]), float(r0["auc_sd"]), nd=3)

                    if method == "aps_fallback":
                        diag = _fmt_pm_math(float(r0["aps_fallback_rate_mean"]), float(r0["aps_fallback_rate_sd"]), nd=3)
                    else:
                        diag = _fmt_pm_math(float(r0["empty_rate_mean"]), float(r0["empty_rate_sd"]), nd=3)

                lines.append(
                    f"{a:.2f} & {pretty_model(model)} & {pretty_method(method)} & {cov} & {cov1} & {cov0} & {size} & {diag} & {auc} \\\\"
                )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved LaTeX key table:", out_path.name)


def write_wdbc_table_full_tex(summ: pd.DataFrame, out_path: Path) -> None:
    """Full appendix table (more columns, same diagnostic definition)."""
    alpha_order = sorted(summ["alpha"].unique())
    model_order = ["logreg", "hgbdt"]
    method_order = ["splitcp", "mondrian", "aps_allowempty", "aps_fallback"]

    lines = []
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}llccccccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Model & Method & $\alpha$ & Emp.\ cov. & cov(M) & cov(B) & Avg.\ size & Diagnostic & Singleton & Ambiguous & Empty(post) & AUC \\")
    lines.append(r"\midrule")

    for model in model_order:
        for method in method_order:
            block = summ[(summ["model"] == model) & (summ["method"] == method)].copy()
            if block.empty:
                continue
            block = block.set_index("alpha")
            first = True
            for a in alpha_order:
                if a not in block.index:
                    continue
                r0 = block.loc[a]

                cov  = _fmt_pm_math(float(r0["coverage_mean"]), float(r0["coverage_sd"]), nd=3)
                cov1 = _fmt_pm_math(float(r0["cov_1_mean"]),    float(r0["cov_1_sd"]),    nd=3)
                cov0 = _fmt_pm_math(float(r0["cov_0_mean"]),    float(r0["cov_0_sd"]),    nd=3)
                size = _fmt_pm_math(float(r0["avg_set_size_mean"]), float(r0["avg_set_size_sd"]), nd=3)
                sing = _fmt_pm_math(float(r0["singleton_rate_mean"]), float(r0["singleton_rate_sd"]), nd=3)
                amb  = _fmt_pm_math(float(r0["ambiguous_rate_mean"]), float(r0["ambiguous_rate_sd"]), nd=3)
                emp_empty = _fmt_pm_math(float(r0["empty_rate_mean"]), float(r0["empty_rate_sd"]), nd=3)
                auc  = _fmt_pm_math(float(r0["auc_mean"]), float(r0["auc_sd"]), nd=3)

                if method == "aps_fallback":
                    diag = _fmt_pm_math(float(r0["aps_fallback_rate_mean"]), float(r0["aps_fallback_rate_sd"]), nd=3)
                else:
                    diag = emp_empty

                model_cell = model if first else ""
                method_cell = method if first else ""
                first = False

                lines.append(
                    f"{model_cell} & {method_cell} & {a:.2f} & {cov} & {cov1} & {cov0} & {size} & {diag} & {sing} & {amb} & {emp_empty} & {auc} \\\\"
                )
            lines.append(r"\midrule")

    if lines and lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}%")
    lines.append(r"}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved LaTeX full table:", out_path.name)


# ============================================================
# Main experiment
# ============================================================

def run_wdbc(
    data_path: str,
    outdir: str,
    seeds: list[int],
    alphas: list[float],
    overwrite: bool = False,
) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    X, y, df = load_wdbc(data_path)
    print("Loaded:", df.shape, "malignant rate:", float(y.mean()))
    print("Outdir:", str(out.resolve()))
    print("Overwrite outputs?", bool(overwrite))

    rows: list[dict] = []
    model_names = ["logreg", "hgbdt"]
    alphas = [float(a) for a in alphas]

    for seed in seeds:
        tr, cal, te = stratified_split_indices(y, seed=seed, train_frac=0.6, cal_frac=0.2)
        Xtr, ytr = X[tr], y[tr]
        Xcal, ycal = X[cal], y[cal]
        Xte, yte = X[te], y[te]

        for model_name in model_names:
            probs_cal, probs_te = fit_model_and_predict_proba(model_name, Xtr, ytr, Xcal, Xte)
            sanity_check_proba(probs_cal, name=f"{model_name}:probs_cal")
            sanity_check_proba(probs_te, name=f"{model_name}:probs_te")

            auc = float(roc_auc_score(yte, probs_te[:, 1]))

            # ---------- SplitCP / Mondrian share the same base score ----------
            cal_scores_split = splitcp_cal_scores(probs_cal, ycal)

            # ---------- APS calibration scores (smoothed) ----------
            base_aps = _make_rng_base(seed, model_name, "aps")
            rng_cal_aps = np.random.default_rng(base_aps + 12_345)
            cal_scores_aps = aps_cal_scores_smoothed(probs_cal, ycal, rng=rng_cal_aps)

            for a in alphas:
                # ============ SplitCP ============
                qhat = conformal_quantile(cal_scores_split, alpha=a)
                sets = predict_sets_threshold(probs_te, qhat=qhat)
                m = set_metrics(sets, yte)
                rows.append({
                    "seed": int(seed), "alpha": float(a), "model": model_name, "method": "splitcp",
                    "qhat": float(qhat), "auc": float(auc),
                    "aps_fallback_rate": float("nan"),
                    **m,
                })

                # ============ Mondrian ============
                qh = mondrian_qhats(cal_scores_split, ycal, alpha=a)
                sets_m = mondrian_predict_sets(probs_te, qhats=qh)
                mm = set_metrics(sets_m, yte)
                rows.append({
                    "seed": int(seed), "alpha": float(a), "model": model_name, "method": "mondrian",
                    "qhat": float("nan"), "auc": float(auc),
                    "aps_fallback_rate": float("nan"),
                    **mm,
                })

                # ============ APS (allow-empty) + APS (+fallback) ============
                qhat_aps = conformal_quantile(cal_scores_aps, alpha=a)
                rng_te = np.random.default_rng(base_aps + 54_321 + 17 * _alpha_key(a))
                sets_raw, raw_empty_mask = aps_predict_raw_sets_smoothed(probs_te, qhat=qhat_aps, rng=rng_te)
                raw_empty_rate = float(raw_empty_mask.mean())

                # allow-empty row
                m_raw = set_metrics(sets_raw, yte)
                rows.append({
                    "seed": int(seed), "alpha": float(a), "model": model_name, "method": "aps_allowempty",
                    "qhat": float(qhat_aps), "auc": float(auc),
                    "aps_fallback_rate": float("nan"),
                    **m_raw,
                })

                # fallback row (derived from raw)
                sets_fb = aps_apply_argmax_fallback(sets_raw, probs_te)
                m_fb = set_metrics(sets_fb, yte)
                rows.append({
                    "seed": int(seed), "alpha": float(a), "model": model_name, "method": "aps_fallback",
                    "qhat": float(qhat_aps), "auc": float(auc),
                    "aps_fallback_rate": raw_empty_rate,  # DIAGNOSTIC for fallback mode
                    **m_fb,
                })

    res = pd.DataFrame(rows)

    # -----------------------------
    # Save raw + summary
    # -----------------------------
    all_path = out / "wdbc_all_results.csv"
    summ_path = out / "wdbc_summary.csv"
    meta_path = out / "metadata.json"
    howto_path = out / "HOWTO_REPRODUCE.txt"

    atomic_write_csv(res, all_path, overwrite=overwrite)
    print("Saved:", all_path.name)

    metrics = [
        "coverage", "cov_1", "cov_0", "avg_set_size",
        "empty_rate", "singleton_rate", "ambiguous_rate",
        "auc", "qhat", "aps_fallback_rate",
    ]
    mean_df = res.groupby(["model", "method", "alpha"])[metrics].mean().add_suffix("_mean")
    std_df = res.groupby(["model", "method", "alpha"])[metrics].std(ddof=1).add_suffix("_sd")
    summ = pd.concat([mean_df, std_df], axis=1).reset_index()

    atomic_write_csv(summ, summ_path, overwrite=overwrite)
    print("Saved:", summ_path.name)

    # -----------------------------
    # Plots (stable filenames for LaTeX)
    # -----------------------------
    plot_with_errorbars(
        summ, out, "coverage", "empirical coverage",
        "WDBC: Coverage vs alpha",
        "wdbc_coverage_curve.png",
        add_target_line=True,
    )
    plot_with_errorbars(
        summ, out, "avg_set_size", "average set size",
        "WDBC: Efficiency vs alpha",
        "wdbc_efficiency_curve.png",
        add_target_line=False,
    )

    # Optional diagnostics: empty-rate for non-fallback; fallback-trigger separately.
    summ_nonfb = summ[summ["method"].isin(["splitcp", "mondrian", "aps_allowempty"])].copy()
    if not summ_nonfb.empty:
        plot_with_errorbars(
            summ_nonfb, out, "empty_rate", "empty-set rate",
            "WDBC: Empty-set rate vs alpha (non-fallback methods)",
            "wdbc_empty_curve.png",
            add_target_line=False,
        )

    summ_fb = summ[summ["method"].isin(["aps_fallback"])].copy()
    if not summ_fb.empty:
        plot_with_errorbars(
            summ_fb, out, "aps_fallback_rate", "fallback-trigger rate (raw empty rate)",
            "WDBC: APS fallback-trigger vs alpha",
            "wdbc_fallback_curve.png",
            add_target_line=False,
        )

    # -----------------------------
    # Tables (stable filenames for LaTeX)
    # -----------------------------
    key_table_path = out / "wdbc_table_2x2_key.tex"
    full_table_path = out / "wdbc_table_2x2_full.tex"
    write_wdbc_table_key_tex(summ, key_table_path)
    write_wdbc_table_full_tex(summ, full_table_path)

    # -----------------------------
    # Metadata + HOWTO
    # -----------------------------
    meta = {
        "timestamp_unix": int(time.time()),
        "data_path": str(Path(data_path).resolve()),
        "outdir": str(out.resolve()),
        "seeds": list(map(int, seeds)),
        "alphas": list(map(float, alphas)),
        "split": {"train": 0.6, "cal": 0.2, "test": 0.2},
        "label_mapping": {"M": 1, "B": 0},
        "methods": [
            "splitcp",
            "mondrian",
            "aps_allowempty",
            "aps_fallback",
        ],
        "diagnostic_definition": {
            "splitcp/mondrian/aps_allowempty": "empty_rate",
            "aps_fallback": "aps_fallback_rate (raw empty rate before fallback; final empty_rate should be 0)",
        },
        "versions": print_versions(),
    }
    atomic_write_text(json.dumps(meta, indent=2), meta_path, overwrite=overwrite)
    print("Saved:", meta_path.name)

    howto = (
        "Reproduce WDBC outputs:\n"
        f"  python wdbc_conformal_2x2.py --data-path \"{Path(data_path).resolve()}\" --outdir \"{out.resolve()}\" --overwrite\n\n"
        "Main outputs:\n"
        "  wdbc_all_results.csv\n"
        "  wdbc_summary.csv\n"
        "  wdbc_table_2x2_key.tex     (LaTeX \\input)\n"
        "  wdbc_table_2x2_full.tex    (appendix)\n"
        "  wdbc_coverage_curve.png\n"
        "  wdbc_efficiency_curve.png\n"
        "  wdbc_empty_curve.png       (non-fallback empty-rate diagnostic)\n"
        "  wdbc_fallback_curve.png    (APS fallback-trigger diagnostic)\n"
        "  metadata.json\n"
    )
    atomic_write_text(howto, howto_path, overwrite=overwrite)
    print("Saved:", howto_path.name)


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=str, required=True, help="Path to wdbc.data")
    ap.add_argument("--outdir", type=str, default="out_wdbc", help="Output directory")
    ap.add_argument("--seeds", type=str, default="1-30", help='e.g. "1-30" or "1,2,3"')
    ap.add_argument("--alphas", type=str, default="0.05,0.10,0.20")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs in outdir.")
    return ap.parse_args()


def parse_seeds(s: str) -> list[int]:
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_alphas(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    args = parse_args()
    run_wdbc(
        data_path=args.data_path,
        outdir=args.outdir,
        seeds=parse_seeds(args.seeds),
        alphas=parse_alphas(args.alphas),
        overwrite=bool(args.overwrite),
    )
