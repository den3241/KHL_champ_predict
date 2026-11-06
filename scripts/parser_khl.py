# scripts/parser_khl.py
"""
Парсер КХЛ: Webcaster mobile API.

Сценарий: прогноз ДО старта сезона → собираем только завершённые сезоны.
Сезоны: 2015/16 – 2025/26, БЕЗ 2019/20 (COVID).

Этапы (stages) берём из data.json → ключ stages_v2:
    [{"id": 370, "khl_id": 1369, "title": "Регулярный чемпионат",
      "type": "regular", "season": "2025/2026", ...}, ...]

Сезон в stages_v2 — строка "YYYY/YYYY". Фильтруем по нашему списку SEASONS
(в формате END year: 2026 == 2025/2026).

Запуск (из корня проекта):
    python -m scripts.parser_khl
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Windows-консоль по умолчанию cp1251 → utf-8 (иначе StreamHandler падает)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ------------------------------------------------------------------
# Настройки
# ------------------------------------------------------------------
BASE_WC = "https://khl.api.webcaster.pro/api/khl_mobile"

# Сезоны в формате END year (2026 = 2025/26).
# 2020 (2019/20) ПРОПУЩЕН — COVID, чемпион не определён.
SEASONS_END_YEAR = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]

# Тот же список, но в строковом виде как в stages_v2: "2015/2016" ... "2025/2026"
SEASONS_LABEL = [f"{y-1}/{y}" for y in SEASONS_END_YEAR]

RAW_DIR = Path("raw")
LOGS_DIR = Path("logs")
RAW_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "parser.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# HTTP-сессия с ретраями
# ------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": "KHL/4.24.0 (iPhone; iOS 16.3; Scale/2.00)",
        "Accept-Language": "ru",
    })
    return s


SESSION = make_session()


def get_json(url: str, params: dict | None = None, pause: float = 1.2) -> Any:
    """GET JSON с паузой между запросами (лимит КХЛ: >= 1 сек)."""
    try:
        r = SESSION.get(url, params=params, timeout=30)
        r.raise_for_status()
        time.sleep(pause)
        return r.json()
    except Exception as e:
        log.error(f"GET {url} params={params} failed: {e}")
        return None


def save_raw(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    log.info(f"saved: {path}")


# ------------------------------------------------------------------
# 1. data.json + извлечение stages_v2
# ------------------------------------------------------------------
def fetch_data_json() -> dict | None:
    data = get_json(f"{BASE_WC}/data.json")
    if data:
        save_raw(data, RAW_DIR / "_meta" / "webcaster_data.json")
    return data


def extract_stages_v2(meta: dict) -> list[dict]:
    """Возвращает list[dict] из meta['stages_v2'] или [] если нет."""
    sv = meta.get("stages_v2")
    if not isinstance(sv, list):
        log.error(f"stages_v2 not found or not list (type={type(sv).__name__})")
        return []
    if sv and not isinstance(sv[0], dict):
        log.error(f"stages_v2 is list[{type(sv[0]).__name__}], expected dict")
        return []
    return sv


def extract_teams(meta: dict) -> list[dict]:
    """
    Объединяем teams и teams_for_filter — это справочник team_id -> name.
    teams_for_filter содержит исторические команды (34 шт.), teams — текущие (22).
    Сохраняем объединённый список в raw/_meta/teams.json.
    """
    teams: list[dict] = []
    for key in ("teams", "teams_for_filter"):
        v = meta.get(key)
        if isinstance(v, list):
            teams.extend(v)
    # дедуп по id
    seen: set[int] = set()
    uniq: list[dict] = []
    for t in teams:
        tid = t.get("id")
        if tid is None or tid in seen:
            continue
        seen.add(tid)
        uniq.append(t)
    return uniq


# ------------------------------------------------------------------
# 2. Игры этапа через Webcaster (events_v2)
# ------------------------------------------------------------------
def fetch_events_for_stage(stage_id: int, season_label: str, stage_type: str) -> list:
    """Забирает все игры этапа с пагинацией. Сохраняет в raw/{end_year}/."""
    season_end_year = int(season_label.split("/")[1])
    page = 1
    all_events: list = []

    while True:
        payload = get_json(
            f"{BASE_WC}/events_v2.json",
            params={"stage_id": stage_id, "page": page},
        )
        if payload is None:
            break
        if isinstance(payload, list):
            events = payload
        elif isinstance(payload, dict):
            events = payload.get("data") or payload.get("events") or []
        else:
            break
        if not events:
            break
        all_events.extend(events)
        log.info(
            f"{season_label} {stage_type} stage={stage_id} page={page}: "
            f"{len(events)} events"
        )
        page += 1
        if len(events) < 16:
            break

    if all_events:
        save_raw(
            all_events,
            RAW_DIR / str(season_end_year) / f"events_{stage_type}.json",
        )
    else:
        log.warning(
            f"{season_label} {stage_type} stage={stage_id}: 0 events"
        )
    return all_events


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main() -> None:
    # 1. Метаданные
    meta = fetch_data_json()
    if meta is None:
        log.error("Не удалось скачать data.json. Останавливаемся.")
        return

    # 1a. Справочник команд
    teams = extract_teams(meta)
    if teams:
        save_raw(teams, RAW_DIR / "_meta" / "teams.json")
        log.info(f"teams saved: {len(teams)}")

    # 1b. Этапы
    stages = extract_stages_v2(meta)
    if not stages:
        log.error("stages_v2 пустой или невалидный. Останавливаемся.")
        return
    log.info(f"stages_v2 loaded: {len(stages)}")

    # 1c. Отбираем только нужные сезоны и только regular/playoff
    wanted = set(SEASONS_LABEL)
    selected = [
        s for s in stages
        if s.get("season") in wanted and s.get("type") in ("regular", "playoff")
    ]
    log.info(
        f"selected stages: {len(selected)} "
        f"(expected ~{len(SEASONS_LABEL) * 2}, минус отсутствующие playoff/regular)"
    )

    # Диагностика: по каким сезонам чего не хватает
    by_season: dict[str, set[str]] = {}
    for s in selected:
        by_season.setdefault(s["season"], set()).add(s["type"])
    for label in SEASONS_LABEL:
        have = by_season.get(label, set())
        missing = {"regular", "playoff"} - have
        if missing:
            log.warning(f"{label}: missing {sorted(missing)}")

    # 2. Скачиваем игры
    for s in selected:
        fetch_events_for_stage(
            stage_id=int(s["id"]),
            season_label=s["season"],
            stage_type=s["type"],
        )

    log.info("Парсинг завершён.")


if __name__ == "__main__":
    main()