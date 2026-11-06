# scripts/tune.py
"""
Grid search гиперпараметров CatBoost на walk-forward.

Сетка:
  depth:            [2, 3, 4]
  learning_rate:    [0.03, 0.05, 0.1]
  l2_leaf_reg:      [3, 10, 30]
  scale_pos_weight: [10, 20]
  → 54 комбинации × 7 фолдов

Защита от overfit на валидацию:
  - маленькая сетка (54, не 500);
  - bootstrap CI для топ-5 конфигураций;
  - выбираем самую простую (min depth, min l2) среди тех,
    чей CI пересекается с CI лучшего.

Метрика: mean walk-forward logloss (softmax по сезону без калибровки).

Выход:
  output/tuning_grid.csv         — все 54 результата
  output/tuning_best.json        — лучшие параметры (для model.py)
  logs/tune.log

Запуск:
    python -m scripts.tune
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
import time
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
    handlers=[
        logging.FileHandler("logs/tune.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

IN_DIR = Path("data/processed")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

MIN_TRAIN_SEASONS = 2
BOOT_N = 2000
SEED = 42

GRID = {
    "depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
    "l2_leaf_reg": [3, 10, 30],
    "scale_pos_weight": [10, 20],
}
FIXED = dict(iterations=300, random_seed=SEED, verbose=0, allow_writing_files=False)


def softmax_by_season(scores: np.ndarray, seasons: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=float)
    for s in np.unique(seasons):
        mask = seasons == s
        x = scores[mask] - scores[mask].max()
        e = np.exp(x)
        out[mask] = e / e.sum()
    return out


def walk_forward_per_fold(df: pd.DataFrame, feat_cols: list[str], params: dict) -> np.ndarray:
    """Возвращает массив потерь по фолдам (len = число фолдов)."""
    years = sorted(df["season_end_year"].unique())
    folds = [y for y in years if sum(1 for x in years if x < y) >= MIN_TRAIN_SEASONS]
    losses = []
    for test_year in folds:
        tr = df[df["season_end_year"] < test_year]
        te = df[df["season_end_year"] == test_year].copy().reset_index(drop=True)
        cb = CatBoostClassifier(**params)
        cb.fit(tr[feat_cols].to_numpy(), tr["is_champion"].to_numpy().astype(int))
        s = cb.predict(te[feat_cols].to_numpy(), prediction_type="RawFormulaVal")
        p = softmax_by_season(s, te["season_end_year"].to_numpy())
        ci = int(te.index[te["is_champion"] == 1][0])
        losses.append(-np.log(p[ci] + 1e-9))
    return np.array(losses, dtype=float)


def bootstrap_ci(losses: np.ndarray, n_boot: int = BOOT_N, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(losses, size=len(losses), replace=True).mean() for _ in range(n_boot)]
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    df = pd.read_parquet(IN_DIR / "features.parquet")
    with open(IN_DIR / "feature_columns_selected.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    log.info(f"data: {len(df)} rows × {len(feat_cols)} features")
    log.info(f"features: {feat_cols}")

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    log.info(f"grid: {len(combos)} combinations")

    results = []
    t0 = time.time()
    for i, vals in enumerate(combos, 1):
        cfg = dict(zip(keys, vals))
        params = {**FIXED, **cfg}
        losses = walk_forward_per_fold(df, feat_cols, params)
        mean, lo, hi = bootstrap_ci(losses)
        results.append({
            **cfg,
            "mean_logloss": float(losses.mean()),
            "ci_lo": lo,
            "ci_hi": hi,
            "per_fold": losses.tolist(),
        })
        if i % 5 == 0 or i == len(combos):
            log.info(f"[{i}/{len(combos)}] elapsed={time.time()-t0:.1f}s last: "
                     f"{cfg} → {losses.mean():.4f}")

    grid_df = pd.DataFrame(results).sort_values("mean_logloss").reset_index(drop=True)
    grid_df["per_fold"] = grid_df["per_fold"].apply(lambda x: ",".join(f"{v:.3f}" for v in x))
    grid_df.to_csv(OUT_DIR / "tuning_grid.csv", index=False)
    log.info(f"saved {OUT_DIR/'tuning_grid.csv'}")

    log.info("\n=== Top-10 configurations ===")
    log.info("\n" + grid_df.drop(columns=["per_fold"]).head(10).round(4).to_string(index=False))

    # --- Выбор: среди top-5 по mean, чей CI пересекается с CI лучшего ---
    best = grid_df.iloc[0]
    best_hi = best["ci_hi"]
    top5 = grid_df.head(5).copy()

    # CI пересекается, если ci_lo кандидата < ci_hi лучшего
    tied = top5[top5["ci_lo"] < best_hi]
    log.info(f"\nconfigs whose 95%CI overlaps with best CI upper bound {best_hi:.3f}:")
    log.info("\n" + tied[["depth", "learning_rate", "l2_leaf_reg", "scale_pos_weight",
                          "mean_logloss", "ci_lo", "ci_hi"]].round(4).to_string(index=False))

    # Простота = меньшая глубина, меньшая l2, меньший scale_pos_weight
    tied_sorted = tied.sort_values(["depth", "l2_leaf_reg", "scale_pos_weight", "learning_rate"],
                                   ascending=[True, True, True, True])
    chosen = tied_sorted.iloc[0]

    log.info("\n=== CHOSEN (simplest among tied) ===")
    log.info(f"depth={int(chosen['depth'])}, lr={chosen['learning_rate']}, "
             f"l2={int(chosen['l2_leaf_reg'])}, spw={int(chosen['scale_pos_weight'])}")
    log.info(f"mean_logloss={chosen['mean_logloss']:.4f}  "
             f"95%CI=[{chosen['ci_lo']:.4f}, {chosen['ci_hi']:.4f}]")

    best_params = {
        **FIXED,
        "depth": int(chosen["depth"]),
        "learning_rate": float(chosen["learning_rate"]),
        "l2_leaf_reg": int(chosen["l2_leaf_reg"]),
        "scale_pos_weight": float(chosen["scale_pos_weight"]),
    }
    with open(OUT_DIR / "tuning_best.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    log.info(f"saved {OUT_DIR/'tuning_best.json'}")


if __name__ == "__main__":
    main()