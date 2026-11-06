# scripts/model_logit.py
"""
Conditional logit (softmax по командам внутри сезона).

Изменения vs предыдущей версии:
  - NaN в фичах импутируются медианой трейна ПЕРЕД стандартизацией
    (CatBoost умеет NaN, scipy/L-BFGS-B — нет).
  - Проверка на NaN в logloss: если модель не сошлась, пропускаем λ.
  - Защита от пустого tied в выборе λ.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/logit.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

IN_DIR = Path("data/processed")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

LAMBDAS = [0.001, 0.01, 0.1, 1.0, 10.0]
MIN_TRAIN_SEASONS = 2
BOOT_N = 2000
SEED = 42


def _impute(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    """Заменить NaN медианой (по столбцам)."""
    X = X.copy()
    mask = np.isnan(X)
    if mask.any():
        X[mask] = np.take(med, np.where(mask)[1])
    return X


def fit_logit(X: np.ndarray, seasons: np.ndarray, champ: np.ndarray, l2: float):
    """
    Возвращает (beta, med, mean, std).
    Порядок: impute медианой → стандартизация → оптимизация NLL с L2.
    """
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)     # если целый столбец NaN
    X = _impute(X, med)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    Xs = (X - mean) / std

    def nll(beta: np.ndarray) -> float:
        scores = Xs @ beta
        loss = 0.0
        for s in np.unique(seasons):
            mask = seasons == s
            sc = scores[mask]
            sc_max = sc.max()
            log_denom = sc_max + np.log(np.exp(sc - sc_max).sum())
            champ_sc = sc[champ[mask] == 1][0]
            loss -= (champ_sc - log_denom)
        loss += 0.5 * l2 * float(np.sum(beta ** 2))
        if not np.isfinite(loss):
            return 1e12
        return loss

    res = minimize(
        nll, np.zeros(X.shape[1]), method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-10, "gtol": 1e-6},
    )
    if not res.success:
        log.warning(f"logit did not converge: {res.message} (nll={res.fun:.4f})")
    return res.x, med, mean, std


def predict_logit(X, seasons, beta, med, mean, std) -> np.ndarray:
    X = _impute(X, med)
    Xs = (X - mean) / std
    scores = Xs @ beta
    p = np.zeros(len(X), dtype=float)
    for s in np.unique(seasons):
        mask = seasons == s
        sc = scores[mask] - scores[mask].max()
        e = np.exp(sc)
        p[mask] = e / e.sum()
    return p


def bootstrap_ci(losses: np.ndarray, n_boot: int = BOOT_N, seed: int = SEED):
    losses = losses[np.isfinite(losses)]
    if len(losses) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(losses, size=len(losses), replace=True).mean() for _ in range(n_boot)]
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def walk_forward(df: pd.DataFrame, feat_cols: list[str], l2: float):
    years = sorted(df["season_end_year"].unique())
    folds = [y for y in years if sum(1 for x in years if x < y) >= MIN_TRAIN_SEASONS]
    losses, oof_rows = [], []
    for test_year in folds:
        tr = df[df["season_end_year"] < test_year]
        te = df[df["season_end_year"] == test_year].copy().reset_index(drop=True)
        beta, med, mean, std = fit_logit(
            tr[feat_cols].to_numpy(),
            tr["season_end_year"].to_numpy(),
            tr["is_champion"].to_numpy().astype(int),
            l2,
        )
        p = predict_logit(te[feat_cols].to_numpy(), te["season_end_year"].to_numpy(),
                          beta, med, mean, std)
        ci = int(te.index[te["is_champion"] == 1][0])
        losses.append(-np.log(p[ci] + 1e-9))
        te["p_logit"] = p
        te["fold_year"] = int(test_year)
        oof_rows.append(te)
    return np.array(losses), pd.concat(oof_rows, ignore_index=True)


def main() -> None:
    df = pd.read_parquet(IN_DIR / "features.parquet")
    with open(IN_DIR / "feature_columns_selected.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    log.info(f"data: {len(df)} rows × {len(feat_cols)} features")
    log.info(f"NaN share per feature (%):\n"
             + (df[feat_cols].isna().mean() * 100).round(1).to_string())

    results, oof_by_lambda = [], {}
    for l2 in LAMBDAS:
        losses, oof = walk_forward(df, feat_cols, l2)
        mean, lo, hi = bootstrap_ci(losses)
        hits1 = hits3 = 0
        for s, g in oof.groupby("season_end_year"):
            g = g.sort_values("p_logit", ascending=False).reset_index(drop=True)
            rank = g.index[g["is_champion"] == 1][0] + 1
            hits1 += int(rank == 1)
            hits3 += int(rank <= 3)
        n_s = oof["season_end_year"].nunique()
        results.append({
            "l2": l2, "mean_logloss": float(np.nanmean(losses)),
            "ci_lo": lo, "ci_hi": hi,
            "precision_at_1": hits1 / n_s, "top3": hits3 / n_s,
            "per_fold": losses.tolist(),
        })
        oof_by_lambda[l2] = oof
        log.info(f"λ={l2:>6}  logloss={np.nanmean(losses):.4f}  "
                 f"95%CI=[{lo:.3f}, {hi:.3f}]  p@1={hits1/n_s:.3f}  top3={hits3/n_s:.3f}")

    grid = pd.DataFrame(results).sort_values("mean_logloss", na_position="last").reset_index(drop=True)
    grid["per_fold"] = grid["per_fold"].apply(
        lambda x: ",".join(f"{v:.3f}" if np.isfinite(v) else "nan" for v in x))
    grid.to_csv(OUT_DIR / "logit_grid.csv", index=False)

    valid = grid[grid["mean_logloss"].notna()]
    if valid.empty:
        log.error("all λ failed — check features / NaN handling")
        return

    chosen = valid.iloc[0]
    chosen_l2 = float(chosen["l2"])
    log.info(f"\nCHOSEN λ={chosen_l2}  logloss={chosen['mean_logloss']:.4f}  "
             f"95%CI=[{chosen['ci_lo']:.3f}, {chosen['ci_hi']:.3f}]  "
             f"p@1={chosen['precision_at_1']:.3f}  top3={chosen['top3']:.3f}")

    tr_all = df[df["season_end_year"] > df["season_end_year"].min()]
    beta_full, med_full, mean_full, std_full = fit_logit(
        tr_all[feat_cols].to_numpy(),
        tr_all["season_end_year"].to_numpy(),
        tr_all["is_champion"].to_numpy().astype(int),
        chosen_l2,
    )
    coefs = pd.DataFrame({"feature": feat_cols, "beta_std": beta_full,
                          "feature_median": med_full,
                          "feature_mean": mean_full, "feature_std": std_full})
    coefs = coefs.reindex(coefs["beta_std"].abs().sort_values(ascending=False).index)
    log.info("\n=== β (std-scaled, abs sorted) ===")
    log.info("\n" + coefs.round(4).to_string(index=False))
    coefs.to_csv(OUT_DIR / "logit_coefficients.csv", index=False)

    with open(OUT_DIR / "logit_best.json", "w", encoding="utf-8") as f:
        json.dump({"l2": chosen_l2, "features": feat_cols,
                   "beta": beta_full.tolist(),
                   "feature_median": med_full.tolist(),
                   "feature_mean": mean_full.tolist(),
                   "feature_std": std_full.tolist()},
                  f, ensure_ascii=False, indent=2)

    oof_best = oof_by_lambda[chosen_l2]
    oof_best.to_parquet(OUT_DIR / "logit_oof.parquet", index=False)
    log.info(f"saved logit_oof.parquet rows={len(oof_best)}")


if __name__ == "__main__":
    main()