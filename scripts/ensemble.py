# scripts/ensemble.py
"""
Ансамбль предсказаний: logit + catboost-tuned + baseline_last_season.

Все модели дают p_champion в OOF (walk-forward, те же 7 фолдов).
Веса перебираем по сетке симплекса (шаг 0.1) → 66 комбинаций.
Критерий выбора: mean walk-forward logloss.

Выход:
  output/ensemble_weights.csv — все 66 комбинаций
  output/ensemble_best.json   — лучшие веса
  output/ensemble_oof.parquet — OOF-предсказания ансамбля
  output/final_comparison.csv — финальная таблица сравнения всех моделей

Запуск:
    python -m scripts.ensemble
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

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
MIN_TRAIN_SEASONS = 2
SEED = 42


def softmax_by_season(scores, seasons):
    out = np.zeros_like(scores, dtype=float)
    for s in np.unique(seasons):
        m = seasons == s
        x = scores[m] - scores[m].max()
        e = np.exp(x); out[m] = e / e.sum()
    return out


def build_oof() -> pd.DataFrame:
    """
    OOF-предсказания трёх моделей по одним и тем же фолдам:
      - logit (из logit_oof.parquet)
      - catboost-tuned (обучаем заново)
      - baseline_last_season (reg_points_pg_l1)
    """
    df = pd.read_parquet(IN_DIR / "features.parquet")
    with open(IN_DIR / "feature_columns_selected.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(OUT_DIR / "tuning_best.json", encoding="utf-8") as f:
        cb_params = json.load(f)

    # --- logit OOF (уже есть) ---
    logit = pd.read_parquet(OUT_DIR / "logit_oof.parquet")
    logit = logit[["season_end_year", "team_id", "team_name",
                   "is_champion", "p_logit"]].rename(columns={"p_logit": "p_logit"})

    # --- CatBoost-tuned OOF ---
    years = sorted(df["season_end_year"].unique())
    folds = [y for y in years if sum(1 for x in years if x < y) >= MIN_TRAIN_SEASONS]
    cb_rows = []
    for test_year in folds:
        tr = df[df["season_end_year"] < test_year]
        te = df[df["season_end_year"] == test_year].copy().reset_index(drop=True)
        cb = CatBoostClassifier(**cb_params)
        cb.fit(tr[feat_cols].to_numpy(), tr["is_champion"].to_numpy().astype(int))
        s = cb.predict(te[feat_cols].to_numpy(), prediction_type="RawFormulaVal")
        te["p_cb"] = softmax_by_season(s, te["season_end_year"].to_numpy())
        cb_rows.append(te[["season_end_year", "team_id", "p_cb"]])
    cb_oof = pd.concat(cb_rows, ignore_index=True)

    # --- baseline_last_season OOF ---
    base_rows = []
    for test_year in folds:
        te = df[df["season_end_year"] == test_year].copy().reset_index(drop=True)
        s = te["reg_points_pg_l1"].fillna(te["reg_points_pg_l1"].median()).to_numpy()
        te["p_base"] = softmax_by_season(s, te["season_end_year"].to_numpy())
        base_rows.append(te[["season_end_year", "team_id", "p_base"]])
    base_oof = pd.concat(base_rows, ignore_index=True)

    merged = logit.merge(cb_oof, on=["season_end_year", "team_id"]).merge(
        base_oof, on=["season_end_year", "team_id"])
    return merged


def eval_ensemble(df: pd.DataFrame, weights: dict[str, float]) -> tuple[float, pd.DataFrame]:
    """Возвращает (mean_logloss, df с p_ens и потерями)."""
    df = df.copy()
    df["p_ens"] = (weights["logit"] * df["p_logit"]
                   + weights["cb"] * df["p_cb"]
                   + weights["base"] * df["p_base"])
    # нормировка на всякий случай (сумма весов = 1 → и так сумма = 1)
    df["p_ens"] = df.groupby("season_end_year")["p_ens"].transform(lambda s: s / s.sum())

    losses = []
    for s, g in df.groupby("season_end_year"):
        ci = int(g.index[g["is_champion"] == 1][0])
        losses.append(-np.log(df.loc[ci, "p_ens"] + 1e-9))
    return float(np.mean(losses)), df


def main() -> None:
    df = build_oof()
    log.info(f"OOF merged: {len(df)} rows, {df['season_end_year'].nunique()} folds")

    # --- 66 весов на симплексе, шаг 0.1 ---
    grid = []
    for i in range(11):
        for j in range(11 - i):
            k = 10 - i - j
            w = {"logit": i / 10, "cb": j / 10, "base": k / 10}
            ll, _ = eval_ensemble(df, w)
            grid.append({**w, "mean_logloss": ll})
    grid_df = pd.DataFrame(grid).sort_values("mean_logloss").reset_index(drop=True)
    grid_df.to_csv(OUT_DIR / "ensemble_weights.csv", index=False)
    log.info("\n=== top-10 ensemble weights ===")
    log.info("\n" + grid_df.head(10).round(4).to_string(index=False))

    # --- Веса отдельных моделей для сравнения ---
    singles = {
        "logit_only":  {"logit": 1.0, "cb": 0.0, "base": 0.0},
        "cb_only":     {"logit": 0.0, "cb": 1.0, "base": 0.0},
        "base_only":   {"logit": 0.0, "cb": 0.0, "base": 1.0},
    }
    singles_ll = {k: eval_ensemble(df, w)[0] for k, w in singles.items()}
    log.info(f"\n=== singles ===\n{singles_ll}")

    # --- Лучший ансамбль ---
    best = grid_df.iloc[0]
    best_w = {"logit": float(best["logit"]),
              "cb": float(best["cb"]),
              "base": float(best["base"])}
    best_ll, best_df = eval_ensemble(df, best_w)
    log.info(f"\nBEST ensemble: {best_w}  logloss={best_ll:.4f}")

    # --- Bootstrap CI (n=7, но всё же) ---
    per_fold = []
    for s, g in best_df.groupby("season_end_year"):
        ci = int(g.index[g["is_champion"] == 1][0])
        per_fold.append(-np.log(best_df.loc[ci, "p_ens"] + 1e-9))
    per_fold = np.array(per_fold)
    rng = np.random.default_rng(SEED)
    boot = [rng.choice(per_fold, len(per_fold), replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    log.info(f"  bootstrap 95%CI = [{lo:.3f}, {hi:.3f}]")

    # --- Итоговая таблица сравнения ---
    final = pd.DataFrame([
        {"model": "logit_only",  "logloss": singles_ll["logit_only"],  "p@1": (best_df.sort_values(['season_end_year','p_logit'],ascending=[True,False]).groupby('season_end_year').head(1)['is_champion'].sum()) / best_df['season_end_year'].nunique()},
        {"model": "catboost_only", "logloss": singles_ll["cb_only"],   "p@1": (best_df.sort_values(['season_end_year','p_cb'],ascending=[True,False]).groupby('season_end_year').head(1)['is_champion'].sum()) / best_df['season_end_year'].nunique()},
        {"model": "baseline_only", "logloss": singles_ll["base_only"], "p@1": (best_df.sort_values(['season_end_year','p_base'],ascending=[True,False]).groupby('season_end_year').head(1)['is_champion'].sum()) / best_df['season_end_year'].nunique()},
        {"model": f"ensemble {best_w}", "logloss": best_ll, "p@1": (best_df.sort_values(['season_end_year','p_ens'],ascending=[True,False]).groupby('season_end_year').head(1)['is_champion'].sum()) / best_df['season_end_year'].nunique()},
    ]).sort_values("logloss").reset_index(drop=True)
    log.info("\n=== FINAL COMPARISON ===\n" + final.round(4).to_string(index=False))
    final.to_csv(OUT_DIR / "final_comparison.csv", index=False)

    with open(OUT_DIR / "ensemble_best.json", "w", encoding="utf-8") as f:
        json.dump({"weights": best_w, "mean_logloss": best_ll,
                   "ci_lo": float(lo), "ci_hi": float(hi)},
                  f, ensure_ascii=False, indent=2)
    best_df.to_parquet(OUT_DIR / "ensemble_oof.parquet", index=False)
    log.info(f"saved {OUT_DIR/'final_comparison.csv'}")


if __name__ == "__main__":
    main()