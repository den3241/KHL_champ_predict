# scripts/interpret.py
"""
SHAP-интерпретация финальной CatBoost-модели (обучена на 2017–2026,
предсказывает 2027).

Выход:
  output/shap_global.csv             — средние |SHAP| по фичам (глобальный рейтинг)
  output/shap_local_top5.csv          — SHAP по 5 командам (Локомотив, Ак Барс, ...)
  output/shap_local_all.parquet       — SHAP всех команд сезона 2027
  output/shap_waterfall_lokomotiv.png — waterfall для Локомотива (опционально)

Запуск:
    python -m scripts.interpret
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
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
FORECAST_SEASON = 2027


def main() -> None:
    df = pd.read_parquet(IN_DIR / "features.parquet")
    with open(IN_DIR / "feature_columns_selected.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(OUT_DIR / "tuning_best.json", encoding="utf-8") as f:
        cb_params = json.load(f)

    train = df[df["season_end_year"] < FORECAST_SEASON].copy()
    test = df[df["season_end_year"] == FORECAST_SEASON].copy().reset_index(drop=True)
    X_tr = train[feat_cols].to_numpy()
    y_tr = train["is_champion"].to_numpy().astype(int)
    X_te = test[feat_cols].to_numpy()

    cb = CatBoostClassifier(**cb_params)
    cb.fit(X_tr, y_tr)

    explainer = shap.TreeExplainer(cb)
    shap_values = explainer.shap_values(X_te)  # (n_te, n_feats)
    base_value = float(explainer.expected_value)

    # --- Глобальный рейтинг ---
    mean_abs = np.abs(shap_values).mean(axis=0)
    global_df = pd.DataFrame({
        "feature": feat_cols, "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    log.info("\n=== Глобальный рейтинг SHAP ===\n" + global_df.round(4).to_string(index=False))
    global_df.to_csv(OUT_DIR / "shap_global.csv", index=False)

    # --- Локальные значения (топ-5 команд 2027) ---
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feat_cols])
    shap_df.insert(0, "team", test["team_name"].to_numpy())
    shap_df.insert(1, "team_id", test["team_id"].to_numpy())
    shap_df.to_parquet(OUT_DIR / "shap_local_all.parquet", index=False)

    top5 = ["Локомотив", "Ак Барс", "Авангард", "ЦСКА", "Металлург Мг"]
    sel = shap_df[shap_df["team"].isin(top5)].copy()
    log.info(f"\n=== SHAP для топ-5 команд (base_value={base_value:.3f}) ===")
    log.info("\n" + sel.set_index("team").round(3).to_string())
    sel.to_csv(OUT_DIR / "shap_local_top5.csv", index=False)

    # --- Вклад фич для Локомотива (отсортированный) ---
    loco = sel[sel["team"] == "Локомотив"].iloc[0].drop(["team", "team_id"])
    loco = loco[loco.apply(lambda x: isinstance(x, (int, float, np.floating, np.integer)))]
    loco = loco.astype(float).sort_values(key=lambda s: s.abs(), ascending=False)
    log.info(f"\n=== Вклад фич в скор Локомотива ===\n{loco.round(4).to_string()}")
    log.info(f"\n=== Вклад фич в скор Локомотива ===\n{loco.round(4).to_string()}")


if __name__ == "__main__":
    main()