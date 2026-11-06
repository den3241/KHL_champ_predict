# scripts/calibrate_predict.py
"""
Калибровка финального прогноза (temperature scaling).

Проблема: CatBoost даёт p=0.98 для фаворита → переуверен.
Решение: подобрать T_m для каждой модели m (logit, catboost) минимизацией
logloss на OOF (walk-forward), затем применить p^(1/T) / Σ p^(1/T)
к предсказаниям 2027.

Обоснование выбора temperature vs isotonic:
  - Temperature scaling — 1 параметр на модель, не переобучается при n=7.
  - Isotonic — непараметрическая, склонна переобучаться при малых данных,
    и не сохраняет порядок/сумму=1 при агрегации по сезону.
  - Наш выход — распределение по сезону, T применяется до нормализации:
    p_cal ∝ p^(1/T), затем сумма = 1 внутри сезона. Чисто и монотонно.

Выход:
  output/predictions_2026_27_calibrated.parquet
  output/calibration_params.json
  output/calibration_comparison.csv
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

OUT_DIR = Path("output")


def temp_softmax(p_raw: np.ndarray, seasons: np.ndarray, T: float) -> np.ndarray:
    """p^(1/T) / Σ, с защитой от log(0)."""
    p_safe = np.clip(p_raw, 1e-12, 1.0)
    log_p = np.log(p_safe) / T
    out = np.zeros_like(p_safe)
    for s in np.unique(seasons):
        m = seasons == s
        x = log_p[m] - log_p[m].max()
        e = np.exp(x)
        out[m] = e / e.sum()
    return out


def logloss(p: np.ndarray, champ_mask: np.ndarray) -> float:
    ci = np.where(champ_mask == 1)[0][0]
    return float(-np.log(p[ci] + 1e-12))


def fit_temperature(oof: pd.DataFrame, prob_col: str) -> float:
    seasons = oof["season_end_year"].to_numpy()
    p_raw = oof[prob_col].to_numpy()
    champ = oof["is_champion"].to_numpy()

    def obj(T):
        losses = []
        for s in np.unique(seasons):
            m = seasons == s
            p = temp_softmax(p_raw[m], np.full(m.sum(), s), T)
            losses.append(logloss(p, champ[m]))
        return float(np.mean(losses))

    res = minimize_scalar(obj, bounds=(0.3, 10.0), method="bounded")
    log.info(f"  {prob_col}: T={res.x:.3f}  logloss={res.fun:.4f} (T=1: {obj(1.0):.4f})")
    return float(res.x)


def main() -> None:
    oof = pd.read_parquet(OUT_DIR / "ensemble_oof.parquet")
    pred = pd.read_parquet(OUT_DIR / "predictions_2026_27.parquet")

    log.info(f"OOF rows={len(oof)}, seasons={sorted(oof['season_end_year'].unique())}")
    log.info("fitting temperatures on OOF ...")
    T_logit = fit_temperature(oof, "p_logit")
    T_cb = fit_temperature(oof, "p_cb")

    # Пересобираем ансамбль 2027 с калибровкой
    seasons_pred = np.array(["2026/2027"] * len(pred))
    p_logit_cal = temp_softmax(pred["p_logit"].to_numpy(), seasons_pred, T_logit)
    p_cb_cal = temp_softmax(pred["p_cb"].to_numpy(), seasons_pred, T_cb)
    p_ens_cal = 0.5 * p_logit_cal + 0.5 * p_cb_cal
    p_ens_cal = p_ens_cal / p_ens_cal.sum()

    out = pred.copy()
    out["p_logit_cal"] = p_logit_cal
    out["p_cb_cal"] = p_cb_cal
    out["p_champion_cal"] = p_ens_cal
    out = out.sort_values("p_champion_cal", ascending=False).reset_index(drop=True)
    out["rank_cal"] = out.index + 1

    log.info("\n=== ДО калибровки (top-10) ===")
    log.info("\n" + out[["rank_in_season", "team", "p_champion"]].head(10).round(4).to_string(index=False))
    log.info("\n=== ПОСЛЕ калибровки (top-10) ===")
    log.info("\n" + out[["rank_cal", "team", "p_champion_cal", "p_logit_cal", "p_cb_cal"]].head(10).round(4).to_string(index=False))

    # Sanity
    assert abs(out["p_champion_cal"].sum() - 1.0) < 1e-6

    final = pd.DataFrame({
        "season": "2026/2027",
        "team": out["team"],
        "p_champion": out["p_champion_cal"],
        "rank_in_season": out["rank_cal"],
    }).sort_values("p_champion", ascending=False).reset_index(drop=True)

    final.to_parquet(OUT_DIR / "predictions_2026_27_calibrated.parquet", index=False)
    final.to_csv(OUT_DIR / "predictions_2026_27_calibrated.csv", index=False)
    log.info(f"saved predictions_2026_27_calibrated.parquet")

    # Сравнение метрик OOF до/после калибровки
    seasons_oof = oof["season_end_year"].to_numpy()
    def oof_logloss(prob_col, T):
        losses = []
        for s in np.unique(seasons_oof):
            m = seasons_oof == s
            p = temp_softmax(oof[prob_col].to_numpy()[m], np.full(m.sum(), s), T)
            losses.append(logloss(p, oof["is_champion"].to_numpy()[m]))
        return float(np.mean(losses))

    with open(OUT_DIR / "calibration_params.json", "w", encoding="utf-8") as f:
        json.dump({"T_logit": T_logit, "T_cb": T_cb,
                   "note": "applied as p^(1/T) then renormalized per season"},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()