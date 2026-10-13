
import math
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Football Model",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = "https://v3.football.api-sports.io"
REQUEST_TIMEOUT = 15
STATUS_LIVE = "LIVE"

st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 1.1rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #111c2d; padding: 12px; border-radius: 12px;
        border: 1px solid #26374d;
    }
    .match-card {
        background: #111c2d; border: 1px solid #26374d;
        border-radius: 14px; padding: 14px; margin: 8px 0;
    }
    .muted {color: #9caec4; font-size: 0.85rem;}
    .score {font-size: 1.4rem; font-weight: 750; text-align: center;}
    div.stButton > button {border-radius: 10px; min-height: 42px;}
    @media (max-width: 640px) {
        .block-container {padding-left: 0.7rem; padding-right: 0.7rem;}
        h1 {font-size: 1.65rem !important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Football Model")
st.caption("Матчи · статистика · вероятности · Telegram")


# ---------------------------------------------------------
# API HELPERS
# ---------------------------------------------------------
def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def api_key() -> str:
    return secret("API_FOOTBALL_KEY")


def api_get(endpoint: str, params: dict | None = None) -> dict:
    """Один HTTP-запрос. Ошибки API показываем пользователю."""
    key = api_key()
    if not key:
        raise RuntimeError(
            "Не найден API_FOOTBALL_KEY. Добавь ключ в Streamlit Secrets."
        )

    try:
        response = requests.get(
            f"{API_BASE}/{endpoint.lstrip('/')}",
            headers={"x-apisports-key": key},
            params=params or {},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ошибка соединения с API-Football: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("API вернул ответ, который не удалось прочитать.") from exc

    errors = payload.get("errors")
    if errors:
        raise RuntimeError(f"API-Football: {errors}")

    return payload


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def get_leagues(_cache_version: str = "v1") -> pd.DataFrame:
    """Справочник текущих лиг. Кэш на 12 часов."""
    payload = api_get("leagues", {"current": "true", "type": "League"})
    rows = []

    for item in payload.get("response", []):
        league = item.get("league") or {}
        country = item.get("country") or {}
        seasons = item.get("seasons") or []

        current_seasons = [s for s in seasons if s.get("current")]
        season_year = (
            current_seasons[0].get("year")
            if current_seasons
            else None
        )

        rows.append(
            {
                "league_id": league.get("id"),
                "league": league.get("name") or "Неизвестная лига",
                "country": country.get("name") or "Мир",
                "country_flag": country.get("flag"),
                "league_logo": league.get("logo"),
                "season": season_year,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=5 * 60, show_spinner=False)
def get_fixtures(day: str, league_id: int | None = None) -> list[dict]:
    """Список матчей на дату. Кэш на 5 минут."""
    params: dict[str, Any] = {"date": day, "timezone": "Europe/Moscow"}
    if league_id is not None:
        params["league"] = int(league_id)

    payload = api_get("fixtures", params)
    return payload.get("response", [])


@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_standings(league_id: int, season: int) -> list[dict]:
    payload = api_get(
        "standings",
        {"league": int(league_id), "season": int(season)},
    )
    response = payload.get("response", [])
    if not response:
        return []

    league = response[0].get("league") or {}
    groups = league.get("standings") or []
    return groups[0] if groups else []


@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_team_last_fixtures(team_id: int, before_day: str, count: int = 5) -> list[dict]:
    """Последние завершённые матчи команды до даты выбранного матча."""
    payload = api_get(
        "fixtures",
        {
            "team": int(team_id),
            "from": (date.fromisoformat(before_day) - timedelta(days=365)).isoformat(),
            "to": before_day,
            "last": int(count),
            "status": "FT",
            "timezone": "Europe/Moscow",
        },
    )
    games = payload.get("response", [])
    games = [g for g in games if (g.get("fixture") or {}).get("status", {}).get("short") in {
        "FT", "AET", "PEN"
    }]
    return games[-count:]


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_h2h(home_id: int, away_id: int, before_day: str) -> list[dict]:
    """Очные встречи до даты матча."""
    payload = api_get(
        "fixtures/headtohead",
        {
            "h2h": f"{int(home_id)}-{int(away_id)}",
            "last": 10,
            "to": before_day,
            "timezone": "Europe/Moscow",
        },
    )
    return payload.get("response", [])


@st.cache_data(ttl=20 * 60, show_spinner=False)
def get_fixture_statistics(fixture_id: int) -> list[dict]:
    payload = api_get("fixtures/statistics", {"fixture": int(fixture_id)})
    return payload.get("response", [])


@st.cache_data(ttl=20 * 60, show_spinner=False)
def get_fixture_odds(fixture_id: int) -> list[dict]:
    payload = api_get("odds", {"fixture": int(fixture_id)})
    return payload.get("response", [])


# ---------------------------------------------------------
# MATCH DATA HELPERS
# ---------------------------------------------------------
def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (ValueError, TypeError):
        return None


def match_goals(game: dict) -> tuple[int | None, int | None]:
    goals = game.get("goals") or {}
    return safe_int(goals.get("home")), safe_int(goals.get("away"))


def fixture_status(game: dict) -> tuple[str, str]:
    fixture = game.get("fixture") or {}
    status = fixture.get("status") or {}
    short = status.get("short") or ""
    elapsed = fixture.get("status", {}).get("elapsed")
    time_text = status.get("long") or short or "Статус неизвестен"

    if short in {"1H", "2H", "HT", "ET", "P", "LIVE", "BT"}:
        time_text = f"LIVE {elapsed or ''}'".strip()
    elif short in {"FT", "AET", "PEN"}:
        time_text = "Завершён"
    elif short in {"NS", "TBD"}:
        kickoff = fixture.get("date")
        if kickoff:
            try:
                time_text = datetime.fromisoformat(
                    kickoff.replace("Z", "+00:00")
                ).strftime("%H:%M")
            except ValueError:
                pass
        else:
            time_text = "Ожидается"

    return short, time_text


def is_finished(game: dict) -> bool:
    short = ((game.get("fixture") or {}).get("status") or {}).get("short")
    return short in {"FT", "AET", "PEN"}


def team_name(game: dict, side: str) -> str:
    return ((game.get("teams") or {}).get(side) or {}).get("name") or "Неизвестная команда"


def team_logo(game: dict, side: str) -> str | None:
    return ((game.get("teams") or {}).get(side) or {}).get("logo")


def show_image(url: str | None, width: int = 38) -> None:
    if url:
        st.image(url, width=width)


def render_match_card(game: dict) -> None:
    fixture = game.get("fixture") or {}
    league = game.get("league") or {}
    home = (game.get("teams") or {}).get("home") or {}
    away = (game.get("teams") or {}).get("away") or {}
    goals_home, goals_away = match_goals(game)
    status, time_text = fixture_status(game)

    with st.container(border=True):
        top = st.columns([0.9, 4.2, 1.2])
        with top[0]:
            show_image(league.get("logo"), 34)
        with top[1]:
            country = league.get("country") or ""
            league_name = league.get("name") or "Турнир"
            flag = league.get("flag")
            if flag:
                st.image(flag, width=20)
            st.caption(f"{country} · {league_name}")
        with top[2]:
            st.caption(time_text)

        c1, c2, c3 = st.columns([3.2, 1.2, 3.2], vertical_alignment="center")
        with c1:
            show_image(home.get("logo"), 34)
            st.write(home.get("name") or "Хозяева")
        with c2:
            if goals_home is None or goals_away is None:
                st.markdown("<div class='score'>– : –</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div class='score'>{goals_home} : {goals_away}</div>",
                    unsafe_allow_html=True,
                )
            st.caption("LIVE" if status in {"1H", "2H", "HT", "ET", "P", "LIVE"} else "")
        with c3:
            show_image(away.get("logo"), 34)
            st.write(away.get("name") or "Гости")

        st.caption(f"Статус: {time_text} · ID матча: {fixture.get('id', '—')}")


# ---------------------------------------------------------
# PROBABILITY MODEL
# ---------------------------------------------------------
def goals_for_side(game: dict, team_id: int) -> tuple[int, int] | None:
    """
    Returns (goals scored by team, goals conceded by team).
    Only completed matches with a known score are accepted.
    """
    if not is_finished(game):
        return None

    hg, ag = match_goals(game)
    if hg is None or ag is None:
        return None

    home_id = safe_int(((game.get("teams") or {}).get("home") or {}).get("id"))
    away_id = safe_int(((game.get("teams") or {}).get("away") or {}).get("id"))

    if home_id == team_id:
        return hg, ag
    if away_id == team_id:
        return ag, hg
    return None


def goals_by_venue(game: dict, team_id: int, venue: str) -> tuple[int, int] | None:
    """venue='home' means the team played at home; venue='away' means away."""
    if not is_finished(game):
        return None

    hg, ag = match_goals(game)
    if hg is None or ag is None:
        return None

    home_id = safe_int(((game.get("teams") or {}).get("home") or {}).get("id"))
    away_id = safe_int(((game.get("teams") or {}).get("away") or {}).get("id"))

    if venue == "home" and home_id == team_id:
        return hg, ag
    if venue == "away" and away_id == team_id:
        return ag, hg
    return None


def mean_or_none(values: list[float]) -> float | None:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def recent_metrics(games: list[dict], team_id: int) -> dict:
    rows = []
    for game in games:
        item = goals_for_side(game, team_id)
        if item:
            scored, conceded = item
            rows.append({"scored": scored, "conceded": conceded})

    return {
        "scored": mean_or_none([r["scored"] for r in rows]),
        "conceded": mean_or_none([r["conceded"] for r in rows]),
        "count": len(rows),
    }


def venue_metrics(games: list[dict], team_id: int, venue: str) -> dict:
    rows = []
    for game in games:
        item = goals_by_venue(game, team_id, venue)
        if item:
            scored, conceded = item
            rows.append({"scored": scored, "conceded": conceded})

    return {
        "scored": mean_or_none([r["scored"] for r in rows]),
        "conceded": mean_or_none([r["conceded"] for r in rows]),
        "count": len(rows),
    }


def h2h_metrics(games: list[dict], home_id: int, away_id: int) -> dict:
    rows = []
    for game in games:
        if not is_finished(game):
            continue
        hg, ag = match_goals(game)
        if hg is None or ag is None:
            continue

        hid = safe_int(((game.get("teams") or {}).get("home") or {}).get("id"))
        aid = safe_int(((game.get("teams") or {}).get("away") or {}).get("id"))

        # Map goals into the selected fixture's home/away order.
        if hid == home_id and aid == away_id:
            rows.append((hg, ag))
        elif hid == away_id and aid == home_id:
            rows.append((ag, hg))

    return {
        "home_scored": mean_or_none([r[0] for r in rows]),
        "away_scored": mean_or_none([r[1] for r in rows]),
        "count": len(rows),
    }


def weighted_average(parts: list[tuple[float | None, float, int]]) -> tuple[float | None, float]:
    """
    parts: (value, weight, sample_count).
    A missing component is skipped and remaining weights are normalized.
    """
    usable = [(v, w) for v, w, n in parts if v is not None and n > 0]
    if not usable:
        return None, 0.0
    total_weight = sum(w for _, w in usable)
    if total_weight <= 0:
        return None, 0.0
    result = sum(v * w for v, w in usable) / total_weight
    coverage = min(1.0, total_weight)
    return float(result), coverage


def poisson_pmf(lam: float, max_goals: int = 10) -> np.ndarray:
    """Stable Poisson probabilities for 0..max_goals."""
    lam = max(0.05, min(float(lam), 6.0))
    values = []
    for k in range(max_goals + 1):
        values.append(math.exp(-lam) * (lam ** k) / math.factorial(k))
    arr = np.array(values, dtype=float)
    # Account for the small probability tail above max_goals.
    arr[-1] += max(0.0, 1.0 - float(arr.sum()))
    return arr / arr.sum()


def calculate_probabilities(lambda_home: float, lambda_away: float) -> dict:
    home_dist = poisson_pmf(lambda_home)
    away_dist = poisson_pmf(lambda_away)
    matrix = np.outer(home_dist, away_dist)

    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    btts = float(matrix[1:, 1:].sum())

    total_dist = np.convolve(home_dist, away_dist)
    totals = {}
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        min_goals = int(line + 0.5)
        over = float(total_dist[min_goals:].sum())
        totals[line] = {"over": over, "under": 1.0 - over}

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "btts_yes": btts,
        "btts_no": 1.0 - btts,
        "totals": totals,
    }


def fit_goal_model(
    home_season: dict,
    away_season: dict,
    home_recent: dict,
    away_recent: dict,
    home_venue: dict,
    away_venue: dict,
    h2h: dict,
    league_home_avg: float = 1.45,
    league_away_avg: float = 1.15,
) -> dict:
    """
    Experimental weighted model:
      - season performance: 40%
      - last five matches: 30%
      - H2H: 10%
      - home/away form: 20%

    For each side, the scoring estimate blends its own goals scored with
    the opponent's goals conceded. Weights are renormalized when data is missing.
    These are starting weights, not proven optimal parameters.
    """
    home_parts = [
        (home_season.get("scored"), 0.20, home_season.get("count", 0)),
        (away_season.get("conceded"), 0.20, away_season.get("count", 0)),
        (home_recent.get("scored"), 0.15, home_recent.get("count", 0)),
        (away_recent.get("conceded"), 0.15, away_recent.get("count", 0)),
        (h2h.get("home_scored"), 0.10, h2h.get("count", 0)),
        (home_venue.get("scored"), 0.10, home_venue.get("count", 0)),
        (away_venue.get("conceded"), 0.10, away_venue.get("count", 0)),
    ]
    away_parts = [
        (away_season.get("scored"), 0.20, away_season.get("count", 0)),
        (home_season.get("conceded"), 0.20, home_season.get("count", 0)),
        (away_recent.get("scored"), 0.15, away_recent.get("count", 0)),
        (home_recent.get("conceded"), 0.15, home_recent.get("count", 0)),
        (h2h.get("away_scored"), 0.10, h2h.get("count", 0)),
        (away_venue.get("scored"), 0.10, away_venue.get("count", 0)),
        (home_venue.get("conceded"), 0.10, home_venue.get("count", 0)),
    ]

    raw_home, home_coverage = weighted_average(home_parts)
    raw_away, away_coverage = weighted_average(away_parts)

    if raw_home is None:
        raw_home = league_home_avg
    if raw_away is None:
        raw_away = league_away_avg

    # Shrink noisy estimates toward league averages.
    # Coverage rises with sample count and available sources.
    lambda_home = league_home_avg * (raw_home / max(league_home_avg, 0.1))
    lambda_away = league_away_avg * (raw_away / max(league_away_avg, 0.1))

    lambda_home = max(0.15, min(lambda_home, 4.5))
    lambda_away = max(0.15, min(lambda_away, 4.5))

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "home_coverage": home_coverage,
        "away_coverage": away_coverage,
        "probabilities": calculate_probabilities(lambda_home, lambda_away),
    }


def percentage(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def stat_value(stats: list[dict], team_id: int, stat_name: str) -> str:
    for item in stats:
        item_team = item.get("team") or {}
        if safe_int(item_team.get("id")) != team_id:
            continue
        for stat in item.get("statistics") or []:
            if str(stat.get("type", "")).lower() == stat_name.lower():
                return str(stat.get("value") if stat.get("value") is not None else "—")
    return "—"


# ---------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------
def send_telegram(text: str) -> tuple[bool, str]:
    token = secret("TELEGRAM_BOT_TOKEN")
    chat_id = secret("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return False, "Добавь TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID в Streamlit Secrets."

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
            timeout=REQUEST_TIMEOUT,
        )
        payload = response.json()
        if response.ok and payload.get("ok"):
            return True, "Сигнал отправлен в Telegram."
        return False, f"Telegram вернул ошибку: {payload.get('description', response.text)}"
    except requests.RequestException as exc:
        return False, f"Не удалось подключиться к Telegram: {exc}"


# ---------------------------------------------------------
# SIDEBAR AND MAIN FIXTURES
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("Настройки")
    selected_date = st.date_input("Дата матчей", value=date.today())
    search_text = st.text_input("Поиск лиги или страны", placeholder="Denmark, Superliga…")
    only_live = st.toggle("Только live", value=False)
    show_all = st.toggle("Показывать все матчи даты", value=True)
    st.caption("Часовой пояс матчей: Europe/Moscow")
    st.caption("API-Football: береги дневной лимит запросов.")

if not api_key():
    st.warning(
        "API-ключ пока не настроен. Добавь API_FOOTBALL_KEY в "
        "Streamlit → Manage app → Settings → Secrets."
    )
    st.stop()

try:
    with st.spinner("Загружаю список турниров…"):
        leagues_df = get_leagues()
except Exception as exc:
    st.error(str(exc))
    st.stop()

if leagues_df.empty:
    st.warning("API не вернул текущие турниры. Проверь ключ и доступность API.")
    st.stop()

if search_text:
    mask = (
        leagues_df["league"].fillna("").str.contains(search_text, case=False, regex=False)
        | leagues_df["country"].fillna("").str.contains(search_text, case=False, regex=False)
    )
    filtered_leagues = leagues_df[mask].copy()
else:
    filtered_leagues = leagues_df.copy()

filtered_leagues = filtered_leagues.sort_values(
    ["country", "league"], na_position="last"
)

league_options = ["Все турниры"] + [
    f"{row.country} — {row.league} (ID {int(row.league_id)})"
    for row in filtered_leagues.itertuples()
]

selected_league_label = st.selectbox(
    "Турнир",
    league_options,
    index=0,
)

selected_league_id = None
selected_league_row = None
if selected_league_label != "Все турниры":
    try:
        selected_league_id = int(selected_league_label.rsplit("ID ", 1)[1].rstrip(")"))
        selected_league_row = filtered_leagues[
            filtered_leagues["league_id"] == selected_league_id
        ].iloc[0]
    except Exception:
        selected_league_id = None
        selected_league_row = None

date_string = selected_date.isoformat()

try:
    with st.spinner("Загружаю матчи…"):
        fixtures = get_fixtures(date_string, selected_league_id)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if only_live:
    fixtures = [
        game for game in fixtures
        if fixture_status(game)[0] in {"1H", "2H", "HT", "ET", "P", "LIVE", "BT"}
    ]

st.subheader("Матчи")
st.caption(f"{selected_date.strftime('%d.%m.%Y')} · Найдено: {len(fixtures)}")

if not fixtures:
    st.info(
        "На выбранную дату матчей не найдено. Попробуй другую дату или выбери «Все турниры». "
        "Покрытие зависит от выбранной лиги и доступного API."
    )
else:
    for game in fixtures:
        render_match_card(game)

st.divider()

# ---------------------------------------------------------
# TABS: ANALYSIS, STANDINGS, TELEGRAM
# ---------------------------------------------------------
tab_analysis, tab_table, tab_telegram, tab_about = st.tabs(
    ["📊 Аналитика", "🏆 Таблица", "📨 Telegram", "ℹ️ О модели"]
)

with tab_analysis:
    st.subheader("Анализ выбранного матча")
    if not fixtures:
        st.info("Сначала выбери дату, на которой есть матчи.")
    else:
        match_labels = []
        match_by_label = {}

        for game in fixtures:
            fixture = game.get("fixture") or {}
            match_id = safe_int(fixture.get("id"))
            label = (
                f"{team_name(game, 'home')} — {team_name(game, 'away')} "
                f"(ID {match_id})"
            )
            match_labels.append(label)
            match_by_label[label] = game

        chosen_label = st.selectbox("Матч", match_labels, key="analysis_match")
        chosen_game = match_by_label[chosen_label]
        chosen_fixture = chosen_game.get("fixture") or {}
        chosen_league = chosen_game.get("league") or {}
        chosen_home = (chosen_game.get("teams") or {}).get("home") or {}
        chosen_away = (chosen_game.get("teams") or {}).get("away") or {}

        home_id = safe_int(chosen_home.get("id"))
        away_id = safe_int(chosen_away.get("id"))
        fixture_id = safe_int(chosen_fixture.get("id"))
        match_day = (chosen_fixture.get("date") or date_string)[:10]
        season = safe_int(chosen_league.get("season"))

        st.write(f"**{chosen_home.get('name')} — {chosen_away.get('name')}**")
        st.caption(
            f"{chosen_league.get('country', '')} · "
            f"{chosen_league.get('name', '')} · сезон {season or '—'}"
        )

        with st.expander("Статистика матча (если доступна)", expanded=False):
            if fixture_id:
                if st.button("Загрузить статистику матча", key="load_stats"):
                    st.session_state["stats_fixture_id"] = fixture_id

                if st.session_state.get("stats_fixture_id") == fixture_id:
                    try:
                        stats = get_fixture_statistics(fixture_id)
                        if not stats:
                            st.info("Статистика для этого матча недоступна.")
                        else:
                            stat_names = [
                                "Shots on Goal",
                                "Total Shots",
                                "Corner Kicks",
                                "Yellow Cards",
                                "Red Cards",
                                "Ball Possession",
                                "Expected Goals",
                            ]
                            for stat_name in stat_names:
                                a, b = st.columns(2)
                                a.metric(
                                    f"{chosen_home.get('name')} · {stat_name}",
                                    stat_value(stats, home_id, stat_name),
                                )
                                b.metric(
                                    f"{chosen_away.get('name')} · {stat_name}",
                                    stat_value(stats, away_id, stat_name),
                                )
                    except Exception as exc:
                        st.error(str(exc))

        if st.button("Рассчитать вероятности", type="primary", key="calculate_model"):
            if not home_id or not away_id:
                st.error("Не удалось определить ID команд.")
            else:
                try:
                    with st.spinner(
                        "Запрашиваю последние матчи команд и очные встречи. "
                        "Это использует несколько API-запросов…"
                    ):
                        home_games = get_team_last_fixtures(home_id, match_day, 5)
                        away_games = get_team_last_fixtures(away_id, match_day, 5)
                        h2h_games = get_h2h(home_id, away_id, match_day)

                    home_recent = recent_metrics(home_games, home_id)
                    away_recent = recent_metrics(away_games, away_id)

                    # Venue form is derived from the returned recent fixtures.
                    home_venue = venue_metrics(home_games, home_id, "home")
                    away_venue = venue_metrics(away_games, away_id, "away")

                    # These are recent-match baselines because the free endpoint
                    # does not guarantee a full season-wide team-stat feed.
                    home_season = recent_metrics(home_games, home_id)
                    away_season = recent_metrics(away_games, away_id)
                    h2h = h2h_metrics(h2h_games, home_id, away_id)

                    model = fit_goal_model(
                        home_season=home_season,
                        away_season=away_season,
                        home_recent=home_recent,
                        away_recent=away_recent,
                        home_venue=home_venue,
                        away_venue=away_venue,
                        h2h=h2h,
                    )
                    st.session_state["last_model"] = {
                        "model": model,
                        "home": chosen_home.get("name"),
                        "away": chosen_away.get("name"),
                        "league": chosen_league.get("name"),
                        "match_id": fixture_id,
                        "match_day": match_day,
                        "home_recent": home_recent,
                        "away_recent": away_recent,
                        "home_venue": home_venue,
                        "away_venue": away_venue,
                        "h2h": h2h,
                    }
                except Exception as exc:
                    st.error(str(exc))

        saved = st.session_state.get("last_model")
        if saved and saved.get("match_id") == fixture_id:
            model = saved["model"]
            probs = model["probabilities"]

            st.markdown("### Ожидаемые голы")
            m1, m2, m3 = st.columns(3)
            m1.metric(saved["home"], f"{model['lambda_home']:.2f}")
            m2.metric(saved["away"], f"{model['lambda_away']:.2f}")
            m3.metric(
                "Ожидаемый тотал",
                f"{model['lambda_home'] + model['lambda_away']:.2f}",
            )

            st.markdown("### Вероятности 1X2")
            c1, c2, c3 = st.columns(3)
            c1.metric("П1", percentage(probs["home_win"]))
            c2.metric("Ничья", percentage(probs["draw"]))
            c3.metric("П2", percentage(probs["away_win"]))

            st.markdown("### Тоталы голов")
            totals_rows = []
            for line, values in probs["totals"].items():
                totals_rows.append(
                    {
                        "Линия": f"{line:.1f}",
                        "Больше": percentage(values["over"]),
                        "Меньше": percentage(values["under"]),
                    }
                )
            st.dataframe(
                pd.DataFrame(totals_rows),
                hide_index=True,
                use_container_width=True,
            )

            c1, c2 = st.columns(2)
            c1.metric("Обе забьют — Да", percentage(probs["btts_yes"]))
            c2.metric("Обе забьют — Нет", percentage(probs["btts_no"]))

            st.markdown("### Форма и размер выборки")
            form_rows = [
                {
                    "Показатель": f"{saved['home']} · последние матчи",
                    "Забито в среднем": saved["home_recent"]["scored"],
                    "Пропущено в среднем": saved["home_recent"]["conceded"],
                    "Матчей": saved["home_recent"]["count"],
                },
                {
                    "Показатель": f"{saved['away']} · последние матчи",
                    "Забито в среднем": saved["away_recent"]["scored"],
                    "Пропущено в среднем": saved["away_recent"]["conceded"],
                    "Матчей": saved["away_recent"]["count"],
                },
                {
                    "Показатель": f"{saved['home']} · дома",
                    "Забито в среднем": saved["home_venue"]["scored"],
                    "Пропущено в среднем": saved["home_venue"]["conceded"],
                    "Матчей": saved["home_venue"]["count"],
                },
                {
                    "Показатель": f"{saved['away']} · в гостях",
                    "Забито в среднем": saved["away_venue"]["scored"],
                    "Пропущено в среднем": saved["away_venue"]["conceded"],
                    "Матчей": saved["away_venue"]["count"],
                },
            ]
            st.dataframe(
                pd.DataFrame(form_rows).round(2),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                f"Очных встреч учтено: {saved['h2h']['count']}. "
                "Из-за лимита бесплатного API показатель «сезон» в этой версии "
                "основан на доступных последних матчах, а не на полном архиве сезона."
            )

            signal_line = st.selectbox(
                "Линия для сообщения",
                [0.5, 1.5, 2.5, 3.5, 4.5],
                index=2,
                key="signal_line",
            )
            signal_side = st.radio(
                "Рынок",
                ["ТБ", "ТМ", "Обе забьют — Да", "П1", "Ничья", "П2"],
                horizontal=True,
                key="signal_side",
            )

            if signal_side == "ТБ":
                signal_probability = probs["totals"][signal_line]["over"]
                market_text = f"ТБ {signal_line:.1f}"
            elif signal_side == "ТМ":
                signal_probability = probs["totals"][signal_line]["under"]
                market_text = f"ТМ {signal_line:.1f}"
            elif signal_side == "Обе забьют — Да":
                signal_probability = probs["btts_yes"]
                market_text = signal_side
            elif signal_side == "П1":
                signal_probability = probs["home_win"]
                market_text = "П1"
            elif signal_side == "Ничья":
                signal_probability = probs["draw"]
                market_text = "Ничья"
            else:
                signal_probability = probs["away_win"]
                market_text = "П2"

            signal_text = (
                f"⚽ Football Model\n"
                f"{saved['league'] or 'Футбол'}\n"
                f"{saved['home']} — {saved['away']}\n"
                f"Рынок: {market_text}\n"
                f"Модельная вероятность: {percentage(signal_probability)}\n"
                f"Ожидаемые голы: {model['lambda_home']:.2f} — "
                f"{model['lambda_away']:.2f}\n"
                f"Матч: {saved['match_day']}\n"
                f"ID: {saved['match_id']}\n"
                f"Это оценка модели, а не гарантия результата."
            )

            with st.expander("Предпросмотр сообщения"):
                st.text(signal_text)

            if st.button("Отправить сигнал в Telegram", key="send_signal"):
                ok, message = send_telegram(signal_text)
                if ok:
                    st.success(message)
                else:
                    st.error(message)

with tab_table:
    st.subheader("Турнирная таблица")
    if selected_league_id is None:
        st.info("Сначала выбери конкретную лигу в фильтре выше.")
    elif selected_league_row is None or pd.isna(selected_league_row["season"]):
        st.warning("Для выбранной лиги не удалось определить текущий сезон.")
    else:
        table_season = int(selected_league_row["season"])
        if st.button("Загрузить таблицу", key="load_table"):
            st.session_state["table_request"] = (
                int(selected_league_id),
                table_season,
            )

        if st.session_state.get("table_request") == (
            int(selected_league_id),
            table_season,
        ):
            try:
                table = get_standings(int(selected_league_id), table_season)
                if not table:
                    st.info("Таблица для этого турнира недоступна.")
                else:
                    rows = []
                    for item in table:
                        team = item.get("team") or {}
                        all_stats = item.get("all") or {}
                        goals = all_stats.get("goals") or {}
                        rows.append(
                            {
                                "Место": item.get("rank"),
                                "Команда": team.get("name"),
                                "И": all_stats.get("played"),
                                "В": all_stats.get("win"),
                                "Н": all_stats.get("draw"),
                                "П": all_stats.get("lose"),
                                "Голы": f"{goals.get('for', 0)}:{goals.get('against', 0)}",
                                "Очки": item.get("points"),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(rows),
                        hide_index=True,
                        use_container_width=True,
                    )
            except Exception as exc:
                st.error(str(exc))

with tab_telegram:
    st.subheader("Подключение Telegram")
    st.write("1. Создай бота через BotFather.")
    st.write("2. Напиши боту /start.")
    st.write("3. Узнай chat_id и добавь токен и chat_id в Streamlit Secrets.")
    st.write("4. Проверь отправку тестового сообщения.")
    if st.button("Отправить тест в Telegram", key="telegram_test"):
        ok, message = send_telegram("⚽ Football Model: тест подключения Telegram.")
        if ok:
            st.success(message)
        else:
            st.error(message)

with tab_about:
    st.subheader("Как работает модель")
    st.markdown(
        """
        **Модель голов основана на распределении Пуассона.**

        Начальные веса для расчёта ожидаемых голов:
        - 40% — сезонная результативность и оборона;
        - 30% — последние пять матчей;
        - 10% — очные встречи;
        - 20% — домашняя форма хозяев и гостевая форма гостей.

        Если компонент не имеет данных, его вес исключается, а оставшиеся веса
        нормализуются. Для каждого показателя используются забитые и пропущенные
        голы, а не только результат победа/ничья/поражение.

        **Ограничения:** это экспериментальная модель, а не проверенный прогнозный
        сервис. В текущей версии «сезонная» часть приближённая: использует доступные
        последние матчи, поскольку бесплатный API имеет дневной лимит. Весовые
        коэффициенты не откалиброваны на историческом тестировании. Вероятности
        не гарантируют прибыль и не учитывают маржу букмекера.

        Для серьёзного использования необходимо сохранить исторические данные,
        проверить модель на матчах, которых она не видела, и отдельно обучить модели
        для угловых, ударов и карточек. Вероятность голов нельзя автоматически
        использовать как вероятность угловых или ударов.
        """
    )

st.divider()
st.caption(
    "Данные предоставляются API-Football. Доступность турниров и статистики зависит "
    "от покрытия источника. Проверяй лимит API и точность модели самостоятельно."
)
