# scripts/features.py
"""
Feature engineering для прогноза чемпиона КХЛ.

Для каждого целевого сезона S и каждой команды, участвующей в S, строим
вектор фич ТОЛЬКО из сезонов < S (никаких утечек).

Особенности:
- Техматчи (is_technical): очки считаем (техпобеда = 3 очка),
  голы — исключаем (5:0 сильно искажает атаку/защиту).
- Учитываем "дыру" 2019/20 (COVID): "последний сезон" — это последний
  ДОСТУПНЫЙ до S, а не строго S-1. Разница попадает в years_since_last_reg.
- Первый целевой сезон — 2017/18 (end_year=2017), потому что для 2016
  нет истории S-1 в наших данных. 2016/17 используется как история.

Выход:
  data/processed/features.parquet       — (season_end_year, team_id) × фичи + is_champion
  data/processed/feature_columns.json   — список имён фич для модели

Запуск:
    python -m scripts.features
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

IN_DIR = Path("data/interim")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Целевые сезоны (end_year). 2016 нет — нечем предсказывать.
TARGET_SEASONS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026, 2027]


# ------------------------------------------------------------------
# Вспомогательные вычисления
# ------------------------------------------------------------------
def compute_playoff_results(games: pd.DataFrame) -> pd.DataFrame:
    """(season_end_year, team_id) → playoff_games, playoff_wins, series_won."""
    po = games[games.stage == "playoff"].copy()

    home = po[["season_end_year", "game_id", "home_team_id", "away_team_id", "winner_team_id"]].copy()
    home = home.rename(columns={"home_team_id": "team_id", "away_team_id": "opp_id"})
    away = po[["season_end_year", "game_id", "away_team_id", "home_team_id", "winner_team_id"]].copy()
    away = away.rename(columns={"away_team_id": "team_id", "home_team_id": "opp_id"})
    long = pd.concat([home, away], ignore_index=True)
    long["is_win"] = (long["team_id"] == long["winner_team_id"]).astype(int)

    series = (
        long.groupby(["season_end_year", "team_id", "opp_id"], as_index=False)
        .agg(series_games=("game_id", "count"), series_wins=("is_win", "sum"))
    )
    series["series_won"] = (series["series_wins"] >= 4).astype(int)

    return (
        series.groupby(["season_end_year", "team_id"], as_index=False)
        .agg(
            playoff_games=("series_games", "sum"),
            playoff_wins=("series_wins", "sum"),
            series_won=("series_won", "sum"),
        )
    )


def compute_regular_goal_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Голы в регулярке БЕЗ техматчей (очки не нужны — они в team_season)."""
    reg = games[(games.stage == "regular") & (~games.is_technical)].copy()

    home = reg[["season_end_year", "home_team_id", "home_score", "away_score"]].copy()
    home.columns = ["season_end_year", "team_id", "goals_for", "goals_against"]
    away = reg[["season_end_year", "away_team_id", "away_score", "home_score"]].copy()
    away.columns = ["season_end_year", "team_id", "goals_for", "goals_against"]
    long = pd.concat([home, away], ignore_index=True)

    return (
        long.groupby(["season_end_year", "team_id"], as_index=False)
        .agg(goals_for=("goals_for", "sum"), goals_against=("goals_against", "sum"))
    )


def build_history_dicts(ts: pd.DataFrame, playoff_res: pd.DataFrame, goal_stats: pd.DataFrame):
    """Возвращает 3 dict'а: team_id → {season_end_year → stats}."""
    reg = ts[ts.stage == "regular"].copy()

    # Голы без техматчей — присоединяем
    reg = reg.drop(columns=["goals_for", "goals_against"]).merge(
        goal_stats, on=["season_end_year", "team_id"], how="left"
    )

    # Число команд в регулярке каждого сезона (для place_pct)
    n_teams = reg.groupby("season_end_year")["team_id"].nunique().to_dict()

    reg_stats: dict[int, dict[int, dict]] = {}
    for _, r in reg.iterrows():
        gf = r["goals_for"] if pd.notna(r["goals_for"]) else 0
        ga = r["goals_against"] if pd.notna(r["goals_against"]) else 0
        games = int(r["games"])
        # игр без техматчей для нормировки голов
        games_no_tech = games - int(r.get("tech_games", 0))
        games_no_tech = max(games_no_tech, 1)

        reg_stats.setdefault(int(r["team_id"]), {})[int(r["season_end_year"])] = {
            "points_pg": r["points"] / games,
            "win_pct": r["win_pct"],
            "goal_diff_pg": (gf - ga) / games_no_tech,
            "goals_for_pg": gf / games_no_tech,
            "goals_against_pg": ga / games_no_tech,
            "place": int(r["regular_place"]),
            "n_teams": n_teams[int(r["season_end_year"])],
        }

    po_stats: dict[int, dict[int, dict]] = {}
    for _, r in playoff_res.iterrows():
        po_stats.setdefault(int(r["team_id"]), {})[int(r["season_end_year"])] = {
            "playoff_games": int(r["playoff_games"]),
            "playoff_wins": int(r["playoff_wins"]),
            "series_won": int(r["series_won"]),
        }
    return reg_stats, po_stats


def most_recent(hist: dict[int, dict], before_year: int) -> tuple[int | None, dict | None]:
    """Последний сезон в hist, который строго меньше before_year."""
    candidates = [y for y in hist if y < before_year]
    if not candidates:
        return None, None
    y = max(candidates)
    return y, hist[y]


def weighted_average(values_by_year: dict[int, float], target_year: int, max_years: int, decay: float = 0.65) -> float:
    num = den = 0.0
    for y, v in values_by_year.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        age = target_year - y
        if age < 1 or age > max_years:
            continue
        w = decay ** (age - 1)
        num += w * v
        den += w
    return num / den if den > 0 else float("nan")


def slope_over_years(values_by_year: dict[int, float], target_year: int, max_years: int) -> float:
    pts = [(y, v) for y, v in values_by_year.items()
           if v is not None and not (isinstance(v, float) and pd.isna(v))
           and 1 <= target_year - y <= max_years]
    if len(pts) < 2:
        return float("nan")
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    return float(np.polyfit(xs, ys, 1)[0])


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main() -> None:
    ts = pd.read_parquet(IN_DIR / "team_season.parquet")
    games = pd.read_parquet(IN_DIR / "games.parquet")
    champs = pd.read_parquet(IN_DIR / "champions.parquet")

    playoff_res = compute_playoff_results(games)
    goal_stats = compute_regular_goal_stats(games)
    reg_stats, po_stats = build_history_dicts(ts, playoff_res, goal_stats)

    # Список команд по сезонам — из регулярки
    reg_only = ts[ts.stage == "regular"]
    teams_by_season: dict[int, list[tuple[int, str]]] = {}
    for S, grp in reg_only.groupby("season_end_year"):
        teams_by_season[int(S)] = list(
            grp[["team_id", "team_name"]].drop_duplicates().itertuples(index=False, name=None)
        )

    # Сезоны каждой команды (для consecutive/total)
    seasons_by_team: dict[int, set[int]] = {}
    for _, r in reg_only.iterrows():
        seasons_by_team.setdefault(int(r["team_id"]), set()).add(int(r["season_end_year"]))

    champs_map = dict(
        zip(champs["season_end_year"].astype(int), champs["champion_team_id"].astype(int))
    )

    # Сезон 2027 = прогноз. Чемпион неизвестен, is_champion = NaN.
    # Список команд для 2027 берём из 2026 (2025/26), т.к. 2026/27 ещё не отыгран.
    FORECAST_SEASON = 2027
    if FORECAST_SEASON not in teams_by_season:
        # копируем команды из 2026
        teams_by_season[FORECAST_SEASON] = teams_by_season.get(2026, [])
    if not teams_by_season[FORECAST_SEASON]:
        raise RuntimeError("нет команд для 2026, не из чего строить прогноз 2027")

    rows: list[dict] = []
    for S in TARGET_SEASONS:
        teams = teams_by_season.get(S, [])
        if not teams:
            log.warning(f"no teams for season {S}")
            continue
        for team_id, team_name in teams:
            team_id = int(team_id)
            reg_hist = reg_stats.get(team_id, {})
            po_hist = po_stats.get(team_id, {})
            row: dict = {"season_end_year": S, "team_id": team_id, "team_name": team_name}

            # --- Регулярка: последний ДОСТУПНЫЙ сезон до S ---
            last_y, last = most_recent(reg_hist, S)
            if last is not None:
                row["reg_points_pg_l1"] = last["points_pg"]
                row["reg_win_pct_l1"] = last["win_pct"]
                row["reg_goal_diff_pg_l1"] = last["goal_diff_pg"]
                row["reg_goals_for_pg_l1"] = last["goals_for_pg"]
                row["reg_goals_against_pg_l1"] = last["goals_against_pg"]
                row["reg_place_l1"] = last["place"]
                row["reg_place_pct_l1"] = last["place"] / last["n_teams"]
                row["years_since_last_reg"] = S - last_y
                row["was_in_league_l1"] = 1
            else:
                for k in [
                    "reg_points_pg_l1", "reg_win_pct_l1", "reg_goal_diff_pg_l1",
                    "reg_goals_for_pg_l1", "reg_goals_against_pg_l1",
                    "reg_place_l1", "reg_place_pct_l1", "years_since_last_reg",
                ]:
                    row[k] = float("nan")
                row["was_in_league_l1"] = 0

            # --- Регулярка: взвешенные средние за 3 и 5 лет ---
            row["reg_points_pg_l3_wavg"] = weighted_average(
                {y: s["points_pg"] for y, s in reg_hist.items()}, S, 3)
            row["reg_win_pct_l3_wavg"] = weighted_average(
                {y: s["win_pct"] for y, s in reg_hist.items()}, S, 3)
            row["reg_goal_diff_pg_l3_wavg"] = weighted_average(
                {y: s["goal_diff_pg"] for y, s in reg_hist.items()}, S, 3)

            row["reg_points_pg_l5_wavg"] = weighted_average(
                {y: s["points_pg"] for y, s in reg_hist.items()}, S, 5)
            row["reg_win_pct_l5_wavg"] = weighted_average(
                {y: s["win_pct"] for y, s in reg_hist.items()}, S, 5)

            # --- Тренд (наклон по последним 5) ---
            row["reg_points_pg_trend5"] = slope_over_years(
                {y: s["points_pg"] for y, s in reg_hist.items()}, S, 5)

            # --- Плей-офф: последний доступный сезон ---
            last_po_y, last_po = most_recent(po_hist, S)
            if last_po is not None:
                row["made_playoff_l1"] = 1
                row["series_won_l1"] = last_po["series_won"]
                pg = last_po["playoff_games"]
                row["playoff_win_pct_l1"] = last_po["playoff_wins"] / pg if pg > 0 else float("nan")
                row["years_since_last_playoff"] = S - last_po_y
            else:
                row["made_playoff_l1"] = 0
                row["series_won_l1"] = 0
                row["playoff_win_pct_l1"] = float("nan")
                row["years_since_last_playoff"] = float("nan")

            # --- Плей-офф: 3-летняя история ---
            sw_l3 = [po_hist.get(y, {}).get("series_won", 0) for y in range(S - 3, S)]
            row["series_won_l3_avg"] = float(np.mean(sw_l3)) if sw_l3 else 0.0
            row["series_won_l3_max"] = int(np.max(sw_l3)) if sw_l3 else 0

            # --- Действующий чемпион ---
            row["is_defending_champion"] = int(champs_map.get(S - 1) == team_id)

            # --- Стабильность в лиге ---
            seasons = seasons_by_team.get(team_id, set())
            cnt, y = 0, S - 1
            while y in seasons:
                cnt += 1
                y -= 1
            row["consecutive_seasons_in_league"] = cnt
            row["total_seasons_in_league"] = len([x for x in seasons if x < S])

            rows.append(row)

    df = pd.DataFrame(rows)
    df["is_champion"] = (df["season_end_year"].map(champs_map) == df["team_id"]).astype(int)
    # 2027 — прогноз, чемпион неизвестен
    df.loc[df["season_end_year"] == FORECAST_SEASON, "is_champion"] = np.nan

    # --- Sanity checks ---
    total_champs = int(df["is_champion"].sum())
    n_known = df[df["is_champion"].notna()]["season_end_year"].nunique()
    expected_known = len([s for s in TARGET_SEASONS if s != FORECAST_SEASON])
    assert total_champs == expected_known, (
        f"expected {expected_known} champions, got {total_champs}"
)
    per_season = df.groupby("season_end_year")["is_champion"].sum()
    if (per_season != 1).any():
        log.warning(f"seasons with !=1 champion:\n{per_season[per_season != 1].to_string()}")

    feature_cols = [
        c for c in df.columns
        if c not in ("season_end_year", "team_id", "team_name", "is_champion")
    ]

    log.info(f"features: {len(feature_cols)} columns × {len(df)} rows")
    log.info(f"per-season champion:")
    log.info(
        "\n" + df[df.is_champion == 1][["season_end_year", "team_name"]].to_string(index=False)
    )
    log.info(f"NaN share per feature (top-5):")
    nan_share = df[feature_cols].isna().mean().sort_values(ascending=False).head(5)
    log.info("\n" + nan_share.round(3).to_string())

    df.to_parquet(OUT_DIR / "features.parquet", index=False)
    with open(OUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    log.info(f"saved {OUT_DIR/'features.parquet'}")
    log.info(f"saved {OUT_DIR/'feature_columns.json'}")


if __name__ == "__main__":
    main()