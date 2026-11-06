# scripts/model.py
"""
Walk-forward валидация прогноза чемпиона КХЛ + финальные предсказания.

Постановка: softmax по командам внутри сезона (listwise), класс = чемпион сезона.
Baselines:
  - baseline_uniform:      p = 1/N на команду;
  - baseline_last_season:  score = reg_points_pg_l1 (лучший в регулярке прошлого сезона).
Модели:
  - catboost_binary:       бинарный классификатор + softmax по сезону;
  - catboost_pairlogit:    CatBoostRanker (PairLogit), group = сезон, listwise.
Калибровка: глобальный temperature scaling T, фитится на out-of-fold предсказаниях
минимизацией logloss.

Выход:
  output/predictions.parquet     — season, team, p_champion, rank_in_season
  output/metrics_by_season.csv   — метрики по каждому фолду для каждой модели
  output/model_comparison.csv    — агрегаты (mean logloss / brier / p@1 / top-3)

Запуск:
    python -m scripts.model
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from scipy.optimize import minimize_scalar

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
OUT_DIR.mkdir(exist_ok=True)

MIN_TRAIN_SEASONS = 2  # для CatBoost нужно хотя бы 2 сезона в трейне
CB_PARAMS = dict(
    iterations=300,
    depth=3,
    learning_rate=0.05,
    l2_leaf_reg=10,
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
)


# ------------------------------------------------------------------
# Softmax / метрики
# ------------------------------------------------------------------
def softmax(x: np.ndarray, T: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / T
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def softmax_by_season(scores: np.ndarray, seasons: np.ndarray, T: float = 1.0) -> np.ndarray:
    out = np.zeros_like(scores, dtype=float)
    for s in np.unique(seasons):
        mask = seasons == s
        out[mask] = softmax(scores[mask], T)
    return out


def per_season_metrics(df: pd.DataFrame, prob_col: str, score_col: str) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("season_end_year"):
        g = g.sort_values(score_col, ascending=False).reset_index(drop=True)
        champ_pos = g.index[g["is_champion"] == 1].tolist()
        if not champ_pos:
            continue
        ci = champ_pos[0]
        p = g[prob_col].to_numpy()
        y = g["is_champion"].to_numpy()
        rank = ci + 1
        rows.append({
            "season_end_year": int(s),
            "season": f"{int(s)-1}/{int(s)}",
            "logloss": float(-np.log(p[ci] + 1e-9)),
            "brier": float(np.sum((p - y) ** 2)),
            "precision_at_1": int(rank == 1),
            "top3": int(rank <= 3),
            "champion_rank": int(rank),
            "champion_prob": float(p[ci]),
            "n_teams": int(len(g)),
        })
    return pd.DataFrame(rows)


def fit_temperature(scores_by_fold: list[np.ndarray], champ_idx_by_fold: list[int]) -> float:
    def logloss(T: float) -> float:
        if T <= 0:
            return 1e9
        total = 0.0
        for scores, ci in zip(scores_by_fold, champ_idx_by_fold):
            p = softmax(scores, T)
            total += -np.log(p[ci] + 1e-9)
        return total / len(scores_by_fold)

    res = minimize_scalar(logloss, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main() -> None:
    df = pd.read_parquet(IN_DIR / "features.parquet")
    with open(IN_DIR / "feature_columns.json", encoding="utf-8") as f:
        feat_cols = json.load(f)

    all_years = sorted(df["season_end_year"].unique())
    folds = [(yrs, y) for y in all_years
             if len(yrs := [x for x in all_years if x < y]) >= MIN_TRAIN_SEASONS]
    log.info(f"target years: {all_years}")
    log.info(f"folds: {len(folds)} → " + ", ".join(f"{te}" for _, te in folds))

    # Храним всё по фолдам
    fold_data: list[dict] = []

    for train_years, test_year in folds:
        tr = df[df["season_end_year"].isin(train_years)].copy()
        te = df[df["season_end_year"] == test_year].copy().reset_index(drop=True)
        X_tr, y_tr = tr[feat_cols].to_numpy(), tr["is_champion"].to_numpy().astype(int)
        X_te = te[feat_cols].to_numpy()

        champ_idx = int(te.index[te["is_champion"] == 1][0])

        # --- Baseline: uniform ---
        p_uniform = np.full(len(te), 1.0 / len(te))

        # --- Baseline: last season reg_points_pg ---
        s_base = te["reg_points_pg_l1"].fillna(te["reg_points_pg_l1"].median()).to_numpy()

        # --- CatBoost binary ---
        cb_bin = CatBoostClassifier(**CB_PARAMS)
        cb_bin.fit(X_tr, y_tr)
        s_bin = cb_bin.predict(X_te, prediction_type="RawFormulaVal")

        # --- CatBoost PairLogit (listwise) ---
        tr_pool = Pool(X_tr, label=y_tr, group_id=tr["season_end_year"].to_numpy())
        cb_rank = CatBoostRanker(loss_function="PairLogit", **CB_PARAMS)
        cb_rank.fit(tr_pool)
        s_rank = cb_rank.predict(X_te)

        fold_data.append({
            "test_year": int(test_year),
            "te": te,
            "champ_idx": champ_idx,
            "scores": {
                "baseline_last_season": s_base,
                "catboost_binary": s_bin,
                "catboost_pairlogit": s_rank,
            },
            "p_uniform": p_uniform,
        })
        log.info(f"fold {test_year}: trained on {len(train_years)} seasons, {len(tr)} rows")

    # --- Калибровка temperature: фитим T глобально по out-of-fold ---
    temps: dict[str, float] = {}
    for name in ("baseline_last_season", "catboost_binary", "catboost_pairlogit"):
        scores_list = [fd["scores"][name] for fd in fold_data]
        champ_list = [fd["champ_idx"] for fd in fold_data]
        temps[name] = fit_temperature(scores_list, champ_list)
        log.info(f"temperature[{name}] = {temps[name]:.3f}")

    # --- Сборка предсказаний по фолдам ---
    metrics_frames: dict[str, pd.DataFrame] = {}
    preds_frames: dict[str, pd.DataFrame] = {}

    # uniform — без калибровки
    p_uniform_df = []
    for fd in fold_data:
        te = fd["te"].copy()
        te["p_champion"] = fd["p_uniform"]
        te["score"] = fd["p_uniform"]
        p_uniform_df.append(te[["season_end_year", "team_id", "team_name", "is_champion",
                                "score", "p_champion"]].copy())
    pu = pd.concat(p_uniform_df, ignore_index=True)
    preds_frames["baseline_uniform"] = pu
    metrics_frames["baseline_uniform"] = per_season_metrics(pu, "p_champion", "score")

    for name in ("baseline_last_season", "catboost_binary", "catboost_pairlogit"):
        T = temps[name]
        rows = []
        for fd in fold_data:
            te = fd["te"].copy()
            scores = fd["scores"][name]
            p = softmax_by_season(scores, te["season_end_year"].to_numpy(), T=T)
            te["score"] = scores
            te["p_champion"] = p
            rows.append(te[["season_end_year", "team_id", "team_name", "is_champion",
                            "score", "p_champion"]].copy())
        combined = pd.concat(rows, ignore_index=True)
        preds_frames[name] = combined
        metrics_frames[name] = per_season_metrics(combined, "p_champion", "score")

    # --- Сводная таблица метрик ---
    summary = []
    for name, m in metrics_frames.items():
        summary.append({
            "model": name,
            "mean_logloss": m["logloss"].mean(),
            "mean_brier": m["brier"].mean(),
            "precision_at_1": m["precision_at_1"].mean(),
            "top3": m["top3"].mean(),
            "mean_champion_rank": m["champion_rank"].mean(),
        })
    summary_df = pd.DataFrame(summary).sort_values("mean_logloss")
    log.info("\n=== model comparison ===\n" + summary_df.round(4).to_string(index=False))

    # --- Определяем победителя по logloss и сохраняем в целевом формате ---
    best_name = summary_df.iloc[0]["model"]
    log.info(f"best model by logloss: {best_name}")

    best = preds_frames[best_name].copy()
    best["rank_in_season"] = (
        best.groupby("season_end_year")["p_champion"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    best["season"] = best["season_end_year"].apply(lambda y: f"{int(y)-1}/{int(y)}")

    # Санити: сумма p_champion по сезону = 1
    sums = best.groupby("season")["p_champion"].sum()
    max_dev = float((sums - 1).abs().max())
    log.info(f"sum(p_champion) per season max deviation: {max_dev:.2e}")
    assert max_dev < 1e-6, f"sum check failed: {sums}"

    output = (
        best[["season", "team_name", "p_champion", "rank_in_season"]]
        .rename(columns={"team_name": "team"})
        .sort_values(["season", "p_champion"], ascending=[True, False])
        .reset_index(drop=True)
    )
    output.to_parquet(OUT_DIR / "predictions.parquet", index=False)
    output.to_csv(OUT_DIR / "predictions.csv", index=False)
    log.info(f"saved {OUT_DIR/'predictions.parquet'} rows={len(output)}")

    # Сохраняем метрики по сезонам и сводку
    all_metrics = pd.concat(
        [m.assign(model=name) for name, m in metrics_frames.items()],
        ignore_index=True,
    )
    all_metrics.to_csv(OUT_DIR / "metrics_by_season.csv", index=False)
    summary_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    log.info(f"saved {OUT_DIR/'metrics_by_season.csv'}")
    log.info(f"saved {OUT_DIR/'model_comparison.csv'}")


if __name__ == "__main__":
    main()