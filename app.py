
import requests
import pandas as pd
import streamlit as st
from datetime import date, timedelta

API_URL = "https://v3.football.api-sports.io"

st.set_page_config(
    page_title="Football Analytics",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football Analytics")
st.caption("Расписание • результаты • форма • статистика • коэффициенты")

# Ключ задаётся через Streamlit Secrets
API_KEY = st.secrets.get("API_FOOTBALL_KEY", "")

if not API_KEY:
    st.error(
        "Не найден API-ключ. Добавь API_FOOTBALL_KEY "
        "в Settings → Secrets в Streamlit."
    )
    st.stop()


class APIError(Exception):
    pass


@st.cache_data(ttl=3600, show_spinner=False)
def api_get(endpoint, params_tuple=()):
    params = dict(params_tuple)
    response = requests.get(
        f"{API_URL}/{endpoint.lstrip('/')}",
        headers={"x-apisports-key": API_KEY},
        params=params,
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    errors = payload.get("errors")
    if errors:
        raise APIError(str(errors))

    return payload.get("response", [])


def get_data(endpoint, params=None):
    params = params or {}
    return api_get(endpoint, tuple(sorted(params.items())))


def safe_get(endpoint, params=None):
    try:
        return get_data(endpoint, params)
    except Exception as exc:
        st.warning(f"Не удалось получить данные: {exc}")
        return []


def table_from_dicts(items):
    if not items:
        st.info("Данные пока недоступны для выбранного турнира или матча.")
        return
    st.dataframe(pd.json_normalize(items), use_container_width=True)


def team_name(team):
    return team.get("name", "Неизвестно")


def match_label(fixture):
    home = team_name(fixture["teams"]["home"])
    away = team_name(fixture["teams"]["away"])
    goals = fixture.get("goals", {})
    home_goals = goals.get("home")
    away_goals = goals.get("away")
    status = fixture.get("fixture", {}).get("status", {})
    elapsed = status.get("elapsed")
    short = status.get("short", "")
    score = (
        f"{home_goals} : {away_goals}"
        if home_goals is not None and away_goals is not None
        else "— : —"
    )
    clock = f" • {elapsed}'" if elapsed is not None else ""
    return f"{home} {score} {away} • {short}{clock}"


# API usage is cached. Reference lists are refreshed once per day.
@st.cache_data(ttl=86400, show_spinner=False)
def get_leagues():
    return api_get("leagues", (("current", "true"),))


@st.cache_data(ttl=3600, show_spinner=False)
def get_fixtures_for_date(day):
    return api_get(
        "fixtures",
        tuple(sorted({
            "date": str(day),
            "timezone": "Europe/Berlin",
        }.items())),
    )


@st.cache_data(ttl=60, show_spinner=False)
def get_live_fixtures():
    return api_get("fixtures", (("live", "all"),))


@st.cache_data(ttl=3600, show_spinner=False)
def get_standings(league_id, season):
    return api_get(
        "standings",
        tuple(sorted({"league": league_id, "season": season}.items())),
    )


@st.cache_data(ttl=1800, show_spinner=False)
def get_team_fixtures(team_id, count=5):
    return api_get(
        "fixtures",
        tuple(sorted({"team": team_id, "last": count}.items())),
    )


@st.cache_data(ttl=120, show_spinner=False)
def get_match_statistics(fixture_id):
    return api_get("fixtures/statistics", (("fixture", fixture_id),))


@st.cache_data(ttl=1800, show_spinner=False)
def get_match_odds(fixture_id):
    return api_get("odds", (("fixture", fixture_id),))


# Load competitions
with st.spinner("Загружаем список соревнований..."):
    leagues_data = safe_get("leagues", {"current": "true"})

if not leagues_data:
    st.error(
        "Список соревнований не загрузился. Проверь API-ключ, "
        "лимит запросов и статус API."
    )
    st.stop()

league_rows = []
for item in leagues_data:
    league = item.get("league", {})
    country = item.get("country", {})
    seasons = item.get("seasons", [])
    current_seasons = [s for s in seasons if s.get("current")]
    season = (
        current_seasons[-1].get("year")
        if current_seasons
        else (seasons[-1].get("year") if seasons else date.today().year)
    )

    league_rows.append({
        "id": league.get("id"),
        "name": league.get("name", "Unknown"),
        "country": country.get("name", "International"),
        "season": season,
        "type": league.get("type", ""),
    })

leagues_df = pd.DataFrame(league_rows).dropna(subset=["id"])
if leagues_df.empty:
    st.error("API не вернул доступные соревнования.")
    st.stop()

# Sidebar filters
st.sidebar.header("Настройки")

countries = ["Все страны"] + sorted(
    leagues_df["country"].dropna().unique().tolist()
)
selected_country = st.sidebar.selectbox("Страна", countries)

filtered = leagues_df.copy()
if selected_country != "Все страны":
    filtered = filtered[filtered["country"] == selected_country]

search = st.sidebar.text_input("Найти лигу или кубок")
if search.strip():
    filtered = filtered[
        filtered["name"].str.contains(search.strip(), case=False, na=False)
    ]

if filtered.empty:
    st.warning("По этому фильтру турниры не найдены.")
    st.stop()

filtered["label"] = (
    filtered["country"] + " — " + filtered["name"]
    + " (" + filtered["season"].astype(str) + ")"
)
selected_label = st.sidebar.selectbox(
    "Турнир",
    filtered["label"].tolist(),
)
selected_league = filtered[filtered["label"] == selected_label].iloc[0]
league_id = int(selected_league["id"])
season = int(selected_league["season"])

st.caption(
    f"Выбран турнир: {selected_league['name']} · "
    f"{selected_league['country']} · сезон {season}"
)

tabs = st.tabs([
    "📅 Матчи",
    "🏆 Таблица",
    "📈 Форма команд",
    "📊 Статистика",
    "💹 Коэффициенты",
])

# Tab 1: schedule, results and live
with tabs[0]:
    st.subheader("Расписание и результаты")

    mode = st.radio(
        "Показать",
        ["По дате", "Лайв", "Матчи выбранной лиги"],
        horizontal=True,
    )

    if mode == "По дате":
        selected_date = st.date_input("Дата", value=date.today())
        with st.spinner("Загружаем матчи..."):
            matches = safe_get(
                "fixtures",
                {
                    "date": str(selected_date),
                    "timezone": "Europe/Berlin",
                },
            )
    elif mode == "Лайв":
        with st.spinner("Загружаем текущие матчи..."):
            matches = safe_get("fixtures", {"live": "all"})
    else:
        start = st.date_input(
            "Начало периода",
            value=date.today(),
            key="start_date",
        )
        end = st.date_input(
            "Конец периода",
            value=date.today() + timedelta(days=7),
            key="end_date",
        )
        if end < start:
            st.error("Конец периода не может быть раньше начала.")
            matches = []
        else:
            matches = safe_get(
                "fixtures",
                {
                    "league": league_id,
                    "season": season,
                    "from": str(start),
                    "to": str(end),
                    "timezone": "Europe/Berlin",
                },
            )

    if matches:
        rows = []
        for m in matches:
            fixture = m.get("fixture", {})
            league = m.get("league", {})
            rows.append({
                "Дата": fixture.get("date", "")[:16].replace("T", " "),
                "Страна": league.get("country", ""),
                "Лига": league.get("name", ""),
                "Матч": match_label(m),
                "Стадион": (fixture.get("venue") or {}).get("name", ""),
                "ID матча": fixture.get("id"),
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Выбери матч по ID в разделе статистики или коэффициентов.")
    else:
        st.info("Матчи не найдены или данные пока не доступны.")

    if st.button("Обновить расписание"):
        st.cache_data.clear()
        st.rerun()

# Tab 2: standings
with tabs[1]:
    st.subheader(f"Таблица: {selected_league['name']}")
    standings_data = safe_get(
        "standings",
        {"league": league_id, "season": season},
    )

    if standings_data:
        league_info = standings_data[0].get("league", {})
        groups = league_info.get("standings", [])
        for index, group in enumerate(groups, start=1):
            if len(groups) > 1:
                st.markdown(f"**Группа {index}**")
            rows = []
            for team in group:
                rows.append({
                    "Место": team.get("rank"),
                    "Команда": (team.get("team") or {}).get("name"),
                    "И": (team.get("all") or {}).get("played"),
                    "В": (team.get("all") or {}).get("win"),
                    "Н": (team.get("all") or {}).get("draw"),
                    "П": (team.get("all") or {}).get("lose"),
                    "Голы": (
                        f"{(team.get('all') or {}).get('goals', {}).get('for', 0)}:"
                        f"{(team.get('all') or {}).get('goals', {}).get('against', 0)}"
                    ),
                    "Разница": team.get("goalsDiff"),
                    "Очки": team.get("points"),
                    "Форма": team.get("form") or "—",
                })
            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("Таблица для этого соревнования недоступна.")

# Tab 3: team form
with tabs[2]:
    st.subheader("Последние матчи команды")

    team_search = st.text_input(
        "Название команды",
        placeholder="Например, Arsenal",
    )

    if team_search.strip() and len(team_search.strip()) >= 3:
        teams = safe_get("teams", {"search": team_search.strip()})
        if teams:
            options = {
                f"{t.get('team', {}).get('name')} — "
                f"{t.get('team', {}).get('country', '')} "
                f"(ID {t.get('team', {}).get('id')})":
                int(t.get("team", {}).get("id"))
                for t in teams
                if t.get("team", {}).get("id") is not None
            }

            if options:
                chosen_team = st.selectbox(
                    "Выбери команду",
                    list(options.keys()),
                )
                team_id = options[chosen_team]

                recent = safe_get(
                    "fixtures",
                    {"team": team_id, "last": 10},
                )

                if recent:
                    result_rows = []
                    for m in recent:
                        goals = m.get("goals", {})
                        home = m.get("teams", {}).get("home", {})
                        away = m.get("teams", {}).get("away", {})
                        hg, ag = goals.get("home"), goals.get("away")
                        team_is_home = home.get("id") == team_id

                        if hg is None or ag is None:
                            result = "Не завершён"
                        else:
                            team_goals = hg if team_is_home else ag
                            opponent_goals = ag if team_is_home else hg
                            result = (
                                "В" if team_goals > opponent_goals
                                else "Н" if team_goals == opponent_goals
                                else "П"
                            )

                        result_rows.append({
                            "Дата": m.get("fixture", {}).get("date", "")[:10],
                            "Турнир": m.get("league", {}).get("name", ""),
                            "Матч": match_label(m),
                            "Итог команды": result,
                        })

                    st.dataframe(
                        pd.DataFrame(result_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("История матчей не найдена.")
        else:
            st.info("Команда не найдена.")
    else:
        st.info("Введи минимум 3 символа названия команды.")

# Tab 4: match statistics
with tabs[3]:
    st.subheader("Статистика конкретного матча")
    fixture_id = st.number_input(
        "ID матча",
        min_value=1,
        step=1,
        value=None,
        placeholder="Введи ID из раздела «Матчи»",
    )

    if st.button("Загрузить статистику", key="stats_button"):
        if fixture_id:
            stats_data = safe_get(
                "fixtures/statistics",
                {"fixture": int(fixture_id)},
            )
            if stats_data:
                rows = []
                for team_stats in stats_data:
                    team = (team_stats.get("team") or {}).get("name", "")
                    for stat in team_stats.get("statistics", []):
                        rows.append({
                            "Команда": team,
                            "Показатель": stat.get("type"),
                            "Значение": stat.get("value"),
                        })
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("У матча нет доступной статистики.")
            else:
                st.info("Статистика отсутствует или ещё не загружена.")
        else:
            st.warning("Сначала введи ID матча.")

# Tab 5: odds
with tabs[4]:
    st.subheader("Коэффициенты букмекеров")
    odds_fixture_id = st.number_input(
        "ID матча для коэффициентов",
        min_value=1,
        step=1,
        value=None,
        placeholder="Введи ID матча",
        key="odds_fixture_id",
    )

    if st.button("Загрузить коэффициенты", key="odds_button"):
        if odds_fixture_id:
            odds_data = safe_get(
                "odds",
                {"fixture": int(odds_fixture_id)},
            )
            rows = []
            for entry in odds_data:
                for bookmaker in entry.get("bookmakers", []):
                    for bet in bookmaker.get("bets", []):
                        for value in bet.get("values", []):
                            rows.append({
                                "Букмекер": bookmaker.get("name"),
                                "Рынок": bet.get("name"),
                                "Исход": value.get("value"),
                                "Коэффициент": value.get("odd"),
                            })

            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "Коэффициенты не найдены. Возможно, для матча "
                    "нет доступного рынка или данные не покрываются тарифом."
                )
        else:
            st.warning("Сначала введи ID матча.")

st.divider()
st.caption(
    "Источник данных: API-Football. Статистика, xG и коэффициенты "
    "зависят от покрытия конкретного матча. Данные API могут обновляться "
    "с задержкой. Модель не гарантирует прибыль."
)
