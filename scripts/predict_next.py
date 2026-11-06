# scripts/predict_next.py
"""
Финальный прогноз на сезон 2026/27.

Обучаем две модели на ВСЕХ 9 известных сезонах (2017–2026, 2026 = 2025/26):
  - conditional logit (λ=1.0, как выбрано в model_logit.py);
  - CatBoost (параметры из tuning_best.json);
Усредняем 50/50 (веса из ensemble.py), получаем p_champion для 22 команд 2027.

Выход:
  output/predictions_2026_27.parquet   — season, team, p_champion, rank_in_season
  output/predictions_2026_27.csv       — то же для глаз
  output/predict_next_meta.json        — параметры, суммы по сезону, число команд

Запуск:
    python -m scripts.predict_next
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

IN_DIR = Path("data/processed")
OUT_DIR = Path("output")
FORECAST_SEASON = 2027
ENSEMBLE_W = {"logit": 0.5, "cb": 0.5}


def _impute(X, med):
    X = X.copy()
    mask = np.isnan(X)
    if mask.any():
        X[mask] = np.take(med, np.where(mask)[1])
    return X


def fit_logit(X, seasons, champ, l2=1.0):
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X = _impute(X, med)
    mean = X.mean(axis=0); std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    Xs = (X - mean) / std

    def nll(beta):
        scores = Xs @ beta
        loss = 0.0
        for s in np.unique(seasons):
            m = seasons == s
            sc = scores[m]; sc_max = sc.max()
            log_denom = sc_max + np.log(np.exp(sc - sc_max).sum())
            champ_sc = sc[champ[m] == 1][0]
            loss -= (champ_sc - log_denom)
        loss += 0.5 * l2 * float(np.sum(beta ** 2))
        return loss if np.isfinite(loss) else 1e12

    res = minimize(nll, np.zeros(X.shape[1]), method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-10, "gtol": 1e-6})
    if not res.success:
        log.warning(f"logit did not converge: {res.message}")
    return res.x, med, mean, std


def predict_logit(X, seasons, beta, med, mean, std):
    X = _impute(X, med)
    Xs = (X - mean) / std
    scores = Xs @ beta
    p = np.zeros(len(X))
    for s in np.unique(seasons):
        m = seasons == s
        sc = scores[m] - scores[m].max()
        e = np.exp(sc)
        p[m] = e / e.sum()
    return p


def softmax_by_season(scores, seasons):
    out = np.zeros_like(scores, dtype=float)
    for s in np.unique(seasons):
        m = seasons == s
        x = scores[m] - scores[m].max()
        e = np.exp(x); out[m] = e / e.sum()
    return out


def main() -> None:
    df = pd.read_parquet(IN_DIR / "features.parquet")
    with open(IN_DIR / "feature_columns_selected.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(OUT_DIR / "tuning_best.json", encoding="utf-8") as f:
        cb_params = json.load(f)

    train = df[df["season_end_year"] < FORECAST_SEASON].copy()
    test = df[df["season_end_year"] == FORECAST_SEASON].copy().reset_index(drop=True)
    log.info(f"train: {len(train)} rows, seasons={sorted(train['season_end_year'].unique())}")
    log.info(f"forecast: {len(test)} teams (2026/27)")

    X_tr = train[feat_cols].to_numpy()
    y_tr = train["is_champion"].to_numpy().astype(int)
    s_tr = train["season_end_year"].to_numpy()
    X_te = test[feat_cols].to_numpy()
    s_te = test["season_end_year"].to_numpy()

    # --- CatBoost ---
    cb = CatBoostClassifier(**cb_params)
    cb.fit(X_tr, y_tr)
    cb_scores = cb.predict(X_te, prediction_type="RawFormulaVal")
    p_cb = softmax_by_season(cb_scores, s_te)

    # --- Logit ---
    beta, med, mean, std = fit_logit(X_tr, s_tr, y_tr, l2=1.0)
    p_logit = predict_logit(X_te, s_te, beta, med, mean, std)

    # --- Ensemble ---
    p_ens = ENSEMBLE_W["logit"] * p_logit + ENSEMBLE_W["cb"] * p_cb
    p_ens = p_ens / p_ens.sum()  # на всякий случай

    out = pd.DataFrame({
        "season": "2026/2027",
        "team_id": test["team_id"].to_numpy(),
        "team": test["team_name"].to_numpy(),
        "p_logit": p_logit,
        "p_cb": p_cb,
        "p_champion": p_ens,
    }).sort_values("p_champion", ascending=False).reset_index(drop=True)
    out["rank_in_season"] = out.index + 1

    # sanity
    ssum = float(out["p_champion"].sum())
    log.info(f"sum(p_champion) = {ssum:.10f}")
    assert abs(ssum - 1.0) < 1e-6, f"sum check failed: {ssum}"

    log.info("\n=== Прогноз чемпиона КХЛ 2026/27 ===")
    log.info("\n" + out[["rank_in_season", "team", "p_champion", "p_logit", "p_cb"]]
             .round(4).to_string(index=False))

    full = out[["season", "team_id", "team", "p_logit", "p_cb",
                "p_champion", "rank_in_season"]].copy()
    full.to_parquet(OUT_DIR / "predictions_2026_27.parquet", index=False)

    # Упрощённая версия (целевой формат проекта)
    final = out[["season", "team", "p_champion", "rank_in_season"]].copy()
    final.to_csv(OUT_DIR / "predictions_2026_27.csv", index=False)
    
    log.info(f"\nsaved {OUT_DIR/'predictions_2026_27.parquet'} rows={len(final)}")

    with open(OUT_DIR / "predict_next_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "forecast_season": "2026/2027",
            "train_seasons": sorted(int(s) for s in train["season_end_year"].unique()),
            "n_train_rows": int(len(train)),
            "n_teams": int(len(test)),
            "ensemble_weights": ENSEMBLE_W,
            "sum_p_champion": ssum,
            "features": feat_cols,
            "logit_beta_std": beta.tolist(),
        }, f, ensure_ascii=False, indent=2)
    log.info(f"saved {OUT_DIR/'predict_next_meta.json'}")


if __name__ == "__main__":
    main()