# scripts/feature_selection.py
"""
Отбор фич для модели чемпиона КХЛ.

Шаги:
  1. Убрать высококоррелированные фичи (|corr| > 0.95) на трейне.
  2. CatBoost feature_importances_ на pooled-обучении (все сезоны) — для ранжирования.
  3. Permutation importance на walk-forward OOF: для каждого фолда обучаем
     CatBoost, считаем падение logloss при перемешивании фичи на тесте.
  4. Кривая logloss(k) — пробуем top-{6,8,10,12,15,24} фич по важности.
  5. Сохраняем лучший набор.

Метрика — mean walk-forward logloss (softmax по сезону, T=1 для честности;
калибровку откладываем до этапа model.py).

Запуск:
    python -m scripts.feature_selection
"""
from __future__ import annotations

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
OUT_DIR.mkdir(exist_ok=True)

MIN_TRAIN_SEASONS = 2
SEED = 42

# Стартовые параметры (с учётом дисбаланса 9/218 → scale_pos_weight ≈ 24)
CB_PARAMS = dict(
    iterations=300,
    depth=3,
    learning_rate=0.05,
    l2_leaf_reg=10,
    scale_pos_weight=20.0,
    random_seed=SEED,
    verbose=0,
    allow_writing_files=False,
)


def softmax_by_season(scores: np.ndarray, seasons: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=float)
    for s in np.unique(seasons):
        mask = seasons == s
        x = scores[mask] - scores[mask].max()
        e = np.exp(x)
        out[mask] = e / e.sum()
    return out


def walk_forward_logloss(
    df: pd.DataFrame,
    feat_cols: list[str],
    params: dict,
    return_oof: bool = False,
) -> tuple[float, pd.DataFrame | None]:
    """Mean walk-forward logloss (softmax по сезону без калибровки)."""
    all_years = sorted(df["season_end_year"].unique())
    folds = [y for y in all_years if sum(1 for x in all_years if x < y) >= MIN_TRAIN_SEASONS]

    fold_losses = []
    oof_rows = []

    for test_year in folds:
        tr = df[df["season_end_year"] < test_year]
        te = df[df["season_end_year"] == test_year].copy().reset_index(drop=True)

        X_tr = tr[feat_cols].to_numpy()
        y_tr = tr["is_champion"].to_numpy().astype(int)
        X_te = te[feat_cols].to_numpy()

        cb = CatBoostClassifier(**params)
        cb.fit(X_tr, y_tr)
        scores = cb.predict(X_te, prediction_type="RawFormulaVal")
        p = softmax_by_season(scores, te["season_end_year"].to_numpy())

        ci = int(te.index[te["is_champion"] == 1][0])
        fold_losses.append(-np.log(p[ci] + 1e-9))

        if return_oof:
            te["_p"] = p
            te["_score"] = scores
            oof_rows.append(te)

    mean_loss = float(np.mean(fold_losses))
    oof = pd.concat(oof_rows, ignore_index=True) if return_oof else None
    return mean_loss, oof


def permutation_importance_oof(
    df: pd.DataFrame,
    feat_cols: list[str],
    params: dict,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """
    Для каждого фолда обучаем модель, считаем дельту logloss
    при перемешивании каждой фичи на тесте. Усредняем по фолдам и повторам.
    """
    all_years = sorted(df["season_end_year"].unique())
    folds = [y for y in all_years if sum(1 for x in all_years if x < y) >= MIN_TRAIN_SEASONS]

    rng = np.random.default_rng(SEED)
    importances: dict[str, list[float]] = {c: [] for c in feat_cols}

    for test_year in folds:
        tr = df[df["season_end_year"] < test_year]
        te = df[df["season_end_year"] == test_year].copy().reset_index(drop=True)
        X_tr = tr[feat_cols].to_numpy()
        y_tr = tr["is_champion"].to_numpy().astype(int)
        X_te = te[feat_cols].to_numpy()

        cb = CatBoostClassifier(**params)
        cb.fit(X_tr, y_tr)
        seasons_te = te["season_end_year"].to_numpy()
        ci = int(te.index[te["is_champion"] == 1][0])

        base_scores = cb.predict(X_te, prediction_type="RawFormulaVal")
        base_p = softmax_by_season(base_scores, seasons_te)
        base_loss = -np.log(base_p[ci] + 1e-9)

        for j, col in enumerate(feat_cols):
            deltas = []
            for _ in range(n_repeats):
                X_perm = X_te.copy()
                rng.shuffle(X_perm[:, j])
                s_perm = cb.predict(X_perm, prediction_type="RawFormulaVal")
                p_perm = softmax_by_season(s_perm, seasons_te)
                loss_perm = -np.log(p_perm[ci] + 1e-9)
                deltas.append(loss_perm - base_loss)
            importances[col].append(float(np.mean(deltas)))

    rows = []
    for col in feat_cols:
        v = np.array(importances[col], dtype=float)
        rows.append({
            "feature": col,
            "perm_imp_mean": float(np.nanmean(v)),
            "perm_imp_std": float(np.nanstd(v)),
            "n_folds": int(len(v)),
        })
    return pd.DataFrame(rows).sort_values("perm_imp_mean", ascending=False).reset_index(drop=True)


def drop_correlated(df: pd.DataFrame, feat_cols: list[str], threshold: float = 0.95) -> list[str]:
    """Жадно убираем фичи с |corr| > threshold (оставляем ту, что раньше в списке)."""
    corr = df[feat_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    kept = [c for c in feat_cols if c not in to_drop]
    if to_drop:
        log.info(f"dropped {len(to_drop)} correlated features: {to_drop}")
    return kept


def main() -> None:
    df = pd.read_parquet(IN_DIR / "features.parquet")
    with open(IN_DIR / "feature_columns.json", encoding="utf-8") as f:
        all_feats = json.load(f)

    log.info(f"loaded {len(df)} rows, {len(all_feats)} features")

    # --- Шаг 1. Корреляционный фильтр ---
    cand_feats = drop_correlated(df, all_feats, threshold=0.95)
    log.info(f"after correlation filter: {len(cand_feats)} features")

    # --- Шаг 2. CatBoost feature_importances_ на pooled (все сезоны, кроме первого) ---
    train_all = df[df["season_end_year"] > df["season_end_year"].min()]
    cb_full = CatBoostClassifier(**CB_PARAMS)
    cb_full.fit(train_all[cand_feats].to_numpy(), train_all["is_champion"].to_numpy().astype(int))
    fi = pd.DataFrame({
        "feature": cand_feats,
        "catboost_importance": cb_full.feature_importances_,
    })
    log.info("\n=== CatBoost feature_importances_ (pooled) ===\n" + fi.sort_values(
        "catboost_importance", ascending=False).round(4).to_string(index=False))

    # --- Шаг 3. Permutation importance на walk-forward ---
    log.info("computing permutation importance on walk-forward OOF (5 repeats per fold)...")
    perm = permutation_importance_oof(df, cand_feats, CB_PARAMS, n_repeats=5)
    log.info("\n=== Permutation importance (OOF) ===\n" + perm.round(4).to_string(index=False))

    merged = fi.merge(perm, on="feature").sort_values("perm_imp_mean", ascending=False).reset_index(drop=True)
    merged.to_csv(OUT_DIR / "feature_selection.csv", index=False)
    log.info(f"saved {OUT_DIR/'feature_selection.csv'}")

    # --- Шаг 4. Кривая logloss(k) ---
    ordered = merged["feature"].tolist()
    ks = [6, 8, 10, 12, 15, len(ordered)]
    ks = sorted(set(min(k, len(ordered)) for k in ks))

    curve = []
    for k in ks:
        cols_k = ordered[:k]
        ll, _ = walk_forward_logloss(df, cols_k, CB_PARAMS)
        curve.append({"k": k, "features": cols_k, "mean_logloss": ll})
        log.info(f"k={k:2d} → mean_logloss={ll:.4f}")

    curve_df = pd.DataFrame([{"k": c["k"], "mean_logloss": c["mean_logloss"]} for c in curve])
    curve_df.to_csv(OUT_DIR / "feature_selection_k_curve.csv", index=False)

    # --- Выбор k: минимум logloss, но при равенстве (в пределах 0.02) — меньший k ---
    best_ll = curve_df["mean_logloss"].min()
    near_tie = curve_df[curve_df["mean_logloss"] <= best_ll + 0.02].sort_values("k")
    best_k = int(near_tie.iloc[0]["k"])
    best_cols = next(c["features"] for c in curve if c["k"] == best_k)

    log.info(f"\nBEST k={best_k}  (logloss={best_ll:.4f})")
    log.info(f"features: {best_cols}")

    with open(IN_DIR / "feature_columns_selected.json", "w", encoding="utf-8") as f:
        json.dump(best_cols, f, ensure_ascii=False, indent=2)
    log.info(f"saved {IN_DIR/'feature_columns_selected.json'}")


if __name__ == "__main__":
    main()