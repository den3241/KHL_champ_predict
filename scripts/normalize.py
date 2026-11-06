# scripts/normalize.py
"""
Нормализация raw/<season>/events_{regular,playoff}.json (Webcaster) в:
  data/interim/games.parquet         — одна строка на матч
  data/interim/team_season.parquet   — агрегаты по (season, stage, team)
  data/interim/champions.parquet     — чемпион каждого сезона

Правила (по диагностике Webcaster):
  - event.game_state_key == "finished"
  - event.type_id == 18           (обычный матч)
  - event.not_regular == False    (не матч звёзд / выставочный)
  - team_a = хозяева, team_b = гости
  - score "2:3"     → обычный матч
  - score "+ : -"   → техпобеда хозяев, 5:0 (КХЛ, п. 5.6)
  - score "- : +"   → техпобеда гостей, 0:5
  - is_so  = scores.bullitt is not None
  - is_ot  = (not is_so) and scores.overtime not in (None, "0:0", "")
  - is_ot_or_so = is_so or is_ot   → для очков КХЛ

Очки КХЛ:
  - победа в осн. время: 3
  - победа в ОТ/Б:       2
  - поражение в ОТ/Б:    1
  - поражение в осн.:    0
  - техническая победа/поражение: как обычная победа/поражение в осн. время

Запуск из корня проекта:
    python -m scripts.normalize
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RAW_DIR = Path("raw")
OUT_DIR = Path("data/interim")
LOGS_DIR = Path("logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "normalize.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SEASON_DIRS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
STAGES = ["regular", "playoff"]
STANDARD_TYPE_ID = 18

TECH_WIN = 5
TECH_LOSS = 0


def _parse_score(s: str | None) -> tuple[int, int, bool] | None:
    """Возвращает (home_score, away_score, is_technical) или None."""
    if not s:
        return None
    s_clean = s.replace(" ", "")
    if ":" in s_clean:
        try:
            a, b = s_clean.split(":", 1)
            return int(a), int(b), False
        except ValueError:
            pass
    if s_clean == "+:-":
        return TECH_WIN, TECH_LOSS, True
    if s_clean == "-:+":
        return TECH_LOSS, TECH_WIN, True
    return None


def _overtime_flags(scores: dict) -> tuple[bool, bool]:
    """(is_ot, is_so) по scores-объекту события."""
    so_raw = scores.get("bullitt")
    ot_raw = scores.get("overtime")
    is_so = so_raw is not None and so_raw != ""
    is_ot = (not is_so) and ot_raw is not None and ot_raw not in ("0:0", "")
    return is_ot, is_so


def load_games() -> pd.DataFrame:
    rows: list[dict] = []
    skipped = {
        "not_finished": 0,
        "non_standard_type": 0,
        "non_regular_flag": 0,
        "bad_score": 0,
        "missing_team": 0,
    }
    non_regular_ids: dict[int, list[int]] = {}
    technical_ids: list[tuple[int, int]] = []  # (season, game_id)

    for season in SEASON_DIRS:
        for stage in STAGES:
            path = RAW_DIR / str(season) / f"events_{stage}.json"
            if not path.exists():
                log.warning(f"missing: {path}")
                continue
            with open(path, encoding="utf-8") as f:
                events = json.load(f)

            for wrapper in events:
                ev = wrapper.get("event") if isinstance(wrapper, dict) else None
                if not ev:
                    continue
                if ev.get("game_state_key") != "finished":
                    skipped["not_finished"] += 1
                    continue
                if ev.get("type_id") != STANDARD_TYPE_ID:
                    skipped["non_standard_type"] += 1
                    continue
                if ev.get("not_regular"):
                    skipped["non_regular_flag"] += 1
                    non_regular_ids.setdefault(season, []).append(ev.get("id"))
                    continue

                score = _parse_score(ev.get("score"))
                if score is None:
                    skipped["bad_score"] += 1
                    log.warning(
                        f"cannot parse score: season={season} stage={stage} "
                        f"game_id={ev.get('id')} score={ev.get('score')!r}"
                    )
                    continue
                hs, aws, is_technical = score
                if is_technical:
                    technical_ids.append((season, ev.get("id")))

                ta, tb = ev.get("team_a") or {}, ev.get("team_b") or {}
                if not ta.get("id") or not tb.get("id"):
                    skipped["missing_team"] += 1
                    continue

                is_ot, is_so = _overtime_flags(ev.get("scores") or {})
                winner_id = ta["id"] if hs > aws else tb["id"]

                rows.append({
                    "season_end_year": season,
                    "season": f"{season-1}/{season}",
                    "stage": stage,
                    "game_id": ev.get("id"),
                    "khl_id": ev.get("khl_id"),
                    "start_at": pd.to_datetime(ev.get("start_at"), unit="ms", utc=True),
                    "home_team_id": ta["id"],
                    "home_team_khl_id": ta.get("khl_id"),
                    "home_team_name": ta.get("name"),
                    "away_team_id": tb["id"],
                    "away_team_khl_id": tb.get("khl_id"),
                    "away_team_name": tb.get("name"),
                    "home_score": hs,
                    "away_score": aws,
                    "raw_score": ev.get("score"),
                    "is_technical": is_technical,
                    "winner_team_id": winner_id,
                    "is_ot": is_ot,
                    "is_so": is_so,
                    "is_ot_or_so": is_ot or is_so,
                    "series_score": ev.get("sscore"),
                })

    log.info(f"skipped: {skipped}")
    if non_regular_ids:
        log.info(f"not_regular games per season: {non_regular_ids}")
    if technical_ids:
        log.info(f"technical games: {len(technical_ids)} → {technical_ids}")

    df = pd.DataFrame(rows)
    log.info(f"games loaded: {len(df)}")
    if not df.empty:
        log.info("by season×stage:\n" + df.groupby(["season", "stage"]).size().to_string())
        log.info(
            "OT/SO share by stage:\n"
            + df.groupby("stage")[["is_ot", "is_so", "is_ot_or_so"]].mean().round(3).to_string()
        )
        log.info(
            "technical by season×stage:\n"
            + df.groupby(["season", "stage"])["is_technical"].sum().to_string()
        )
    return df


def build_team_season(games: pd.DataFrame) -> pd.DataFrame:
    """Агрегаты по (season, stage, team)."""
    base_cols = [
        "season_end_year", "season", "stage", "game_id",
        "home_team_id", "home_team_name", "home_score", "away_score",
        "winner_team_id", "is_ot", "is_so", "is_ot_or_so", "is_technical",
    ]
    out_cols = [
        "season_end_year", "season", "stage", "game_id",
        "team_id", "team_name", "goals_for", "goals_against",
        "winner_team_id", "is_ot", "is_so", "is_ot_or_so", "is_technical",
    ]

    home = games[base_cols].copy()
    home.columns = out_cols
    home["is_home"] = 1

    away = games[[
        "season_end_year", "season", "stage", "game_id",
        "away_team_id", "away_team_name", "away_score", "home_score",
        "winner_team_id", "is_ot", "is_so", "is_ot_or_so", "is_technical",
    ]].copy()
    away.columns = out_cols
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True)
    long["is_win"] = (long["team_id"] == long["winner_team_id"]).astype(int)

    def khl_points(r) -> int:
        if r["is_win"]:
            return 2 if r["is_ot_or_so"] else 3
        return 1 if r["is_ot_or_so"] else 0

    long["points"] = long.apply(khl_points, axis=1)

    agg = (
        long.groupby(
            ["season_end_year", "season", "stage", "team_id", "team_name"],
            as_index=False,
        )
        .agg(
            games=("game_id", "count"),
            wins=("is_win", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            points=("points", "sum"),
            ot_games=("is_ot", "sum"),
            so_games=("is_so", "sum"),
            tech_games=("is_technical", "sum"),
        )
    )
    agg["win_pct"] = agg["wins"] / agg["games"]
    agg["goal_diff"] = agg["goals_for"] - agg["goals_against"]
    agg["goals_for_pg"] = agg["goals_for"] / agg["games"]
    agg["goals_against_pg"] = agg["goals_against"] / agg["games"]

    reg = agg[agg["stage"] == "regular"].copy()
    reg = reg.sort_values(
        ["season_end_year", "points", "goal_diff"],
        ascending=[True, False, False],
    )
    reg["regular_place"] = reg.groupby("season_end_year").cumcount() + 1
    place_map = reg.set_index(["season_end_year", "team_id"])["regular_place"]

    agg["regular_place"] = agg.apply(
        lambda r: place_map.get((r["season_end_year"], r["team_id"]))
        if r["stage"] == "regular" else None,
        axis=1,
    )
    return agg


def build_champions(games: pd.DataFrame) -> pd.DataFrame:
    """Чемпион = победитель последнего по времени реального (не тех) матча плей-офф."""
    po = games[(games["stage"] == "playoff") & (~games["is_technical"])].copy()
    if po.empty:
        log.error("no real playoff games, cannot determine champions")
        return pd.DataFrame()

    po = po.sort_values(["season_end_year", "start_at"])
    rows = []
    for season, grp in po.groupby("season_end_year"):
        last = grp.iloc[-1]
        champ_id = last["winner_team_id"]
        champ_name = (
            last["home_team_name"] if champ_id == last["home_team_id"]
            else last["away_team_name"]
        )
        ss = last.get("series_score")
        if isinstance(ss, str) and ":" in ss:
            a, b = ss.split(":", 1)
            max_wins = max(int(a), int(b))
            if max_wins != 4:
                log.warning(
                    f"season {season}: final series_score={ss}, "
                    f"expected 4 on winner side (game_id={last['game_id']})"
                )
        else:
            log.warning(
                f"season {season}: no series_score for final (game_id={last['game_id']})"
            )

        rows.append({
            "season_end_year": season,
            "season": last["season"],
            "champion_team_id": champ_id,
            "champion_team_name": champ_name,
            "final_game_date": last["start_at"],
            "final_game_id": last["game_id"],
            "final_series_score": ss,
        })
    df = pd.DataFrame(rows).sort_values("season_end_year").reset_index(drop=True)
    log.info(
        "champions:\n"
        + df[["season", "champion_team_name", "final_series_score"]].to_string(index=False)
    )
    return df


def main() -> None:
    games = load_games()
    if games.empty:
        log.error("no games loaded, abort")
        return
    games.to_parquet(OUT_DIR / "games.parquet", index=False)
    log.info(f"saved {OUT_DIR/'games.parquet'} rows={len(games)}")

    ts = build_team_season(games)
    ts.to_parquet(OUT_DIR / "team_season.parquet", index=False)
    log.info(f"saved {OUT_DIR/'team_season.parquet'} rows={len(ts)}")

    ch = build_champions(games)
    if not ch.empty:
        ch.to_parquet(OUT_DIR / "champions.parquet", index=False)
        log.info(f"saved {OUT_DIR/'champions.parquet'} rows={len(ch)}")


if __name__ == "__main__":
    main()