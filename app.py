import math
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import log_loss

st.set_page_config(
page_title="Футбольная аналитика",
page_icon="⚽",
layout="centered",
initial_sidebar_state="collapsed",
)

st.title("⚽ Футбольная аналитика")
st.caption("Вероятности исходов, тоталы, обе забьют и исторический бэктест")

REQUIRED = ["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]

def load_matches(file):
df = pd.read_csv(file)
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
raise ValueError("Нет колонок: " + ", ".join(missing))

df = df[REQUIRED].copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["HomeTeam"] = df["HomeTeam"].astype("string").str.strip()
df["AwayTeam"] = df["AwayTeam"].astype("string").str.strip()
df["HomeGoals"] = pd.to_numeric(df["HomeGoals"], errors="coerce")
df["AwayGoals"] = pd.to_numeric(df["AwayGoals"], errors="coerce")
df = df.dropna(subset=REQUIRED)
df = df[
    (df["HomeTeam"] != "")
    & (df["AwayTeam"] != "")
    & (df["HomeGoals"] >= 0)
    & (df["AwayGoals"] >= 0)
    & (df["HomeGoals"] % 1 == 0)
    & (df["AwayGoals"] % 1 == 0)
]
df["HomeGoals"] = df["HomeGoals"].astype(int)
df["AwayGoals"] = df["AwayGoals"].astype(int)
df = df.sort_values("Date").reset_index(drop=True)

if len(df) < 20:
    raise ValueError("После очистки осталось меньше 20 матчей.")
return df

def fit_ratings(history, smoothing=5):
avg_home = float(history["HomeGoals"].mean())
avg_away = float(history["AwayGoals"].mean())

if avg_home <= 0 or avg_away <= 0:
    avg_home, avg_away = 1.35, 1.10

teams = set(history["HomeTeam"]) | set(history["AwayTeam"])
ratings = {}

for team in teams:
    h = history[history["HomeTeam"] == team]
    a = history[history["AwayTeam"] == team]

    home_scored = (h["HomeGoals"].sum() + smoothing * avg_home) / (len(h) + smoothing)
    home_conceded = (h["AwayGoals"].sum() + smoothing * avg_away) / (len(h) + smoothing)
    away_scored = (a["AwayGoals"].sum() + smoothing * avg_away) / (len(a) + smoothing)
    away_conceded = (a["HomeGoals"].sum() + smoothing * avg_home) / (len(a) + smoothing)

    ratings[team] = {
        "ha": home_scored / avg_home,
        "hd": home_conceded / avg_away,
        "aa": away_scored / avg_away,
        "ad": away_conceded / avg_home,
    }

return ratings, avg_home, avg_away

def predict(home, away, ratings, avg_home, avg_away, max_goals=15):
h = ratings.get(home, {})
a = ratings.get(away, {})

xg_home = max(0.05, min(5.5, h.get("ha", 1) * a.get("ad", 1) * avg_home))
xg_away = max(0.05, min(5.5, a.get("aa", 1) * h.get("hd", 1) * avg_away))

hp = poisson.pmf(np.arange(max_goals + 1), xg_home)
ap = poisson.pmf(np.arange(max_goals + 1), xg_away)
hp[-1] += max(0, 1 - hp.sum())
ap[-1] += max(0, 1 - ap.sum())
matrix = np.outer(hp, ap)

p_home = float(np.tril(matrix, -1).sum())
p_draw = float(np.trace(matrix))
p_away = float(np.triu(matrix, 1).sum())
btts = float(matrix[1:, 1:].sum())

totals = {}
for line in (1.5, 2.5, 3.5):
    over = float(sum(
        matrix[i, j]
        for i in range(max_goals + 1)
        for j in range(max_goals + 1)
        if i + j > line
    ))
    totals[f"ТБ {line}"] = over
    totals[f"ТМ {line}"] = 1 - over

result = {
    "xG хозяев": xg_home,
    "xG гостей": xg_away,
    "Победа хозяев": p_home,
    "Ничья": p_draw,
    "Победа гостей": p_away,
    "Обе забьют — Да": btts,
    "Обе забьют — Нет": 1 - btts,
}
result.update(totals)
return result

def percent(p):
return f"{p * 100:.1f}%"

st.subheader("1. Исторические данные")
st.write("CSV должен содержать Date, HomeTeam, AwayTeam, HomeGoals, AwayGoals.")
st.caption("Дата в формате YYYY-MM-DD. Нужны завершённые матчи.")

template = pd.DataFrame({
"Date": ["2026-08-15", "2026-08-16"],
"HomeTeam": ["Arsenal", "Chelsea"],
"AwayTeam": ["Man City", "Liverpool"],
"HomeGoals": [2, 1],
"AwayGoals": [1, 1],
})
st.download_button(
"Скачать пример CSV",
template.to_csv(index=False).encode("utf-8"),
"matches_template.csv",
"text/csv",
)

uploaded = st.file_uploader("Загрузить исторические матчи", type=["csv"])

if "matches" not in st.session_state:
st.session_state.matches = None

if uploaded is not None:
try:
st.session_state.matches = load_matches(uploaded)
st.success(f"Загружено матчей: {len(st.session_state.matches)}")
except Exception as error:
st.session_state.matches = None
st.error(f"Ошибка CSV: {error}")

df = st.session_state.matches

if df is not None:
st.write(f"Период: {df.Date.min().date()} — {df.Date.max().date()}")
st.dataframe(df.tail(8), use_container_width=True, hide_index=True)

st.subheader("2. Прогноз матча")
teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("Хозяева", teams, index=0)
with col2:
    away = st.selectbox("Гости", teams, index=min(1, len(teams) - 1))

if home == away:
    st.error("Выбери две разные команды.")
else:
    ratings, avg_home, avg_away = fit_ratings(df)
    result = predict(home, away, ratings, avg_home, avg_away)

    st.markdown(f"### {home} — {away}")
    c1, c2 = st.columns(2)
    c1.metric("Ожидаемые голы хозяев", f"{result['xG хозяев']:.2f}")
    c2.metric("Ожидаемые голы гостей", f"{result['xG гостей']:.2f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Победа хозяев", percent(result["Победа хозяев"]))
    c2.metric("Ничья", percent(result["Ничья"]))
    c3.metric("Победа гостей", percent(result["Победа гостей"]))

    c1, c2 = st.columns(2)
    c1.metric("ТБ 2.5", percent(result["ТБ 2.5"]))
    c2.metric("Обе забьют — Да", percent(result["Обе забьют — Да"]))

    st.subheader("3. Сравнение с коэффициентом")
    market_options = [k for k in result if k not in ("xG хозяев", "xG гостей")]
    market = st.selectbox("Рынок", market_options)
    odds = st.number_input(
        "Коэффициент букмекера",
        min_value=1.01,
        max_value=1000.0,
        value=1.80,
        step=0.01,
    )
    probability = result[market]
    fair_odds = 1 / probability if probability > 0 else float("inf")
    ev = (probability * odds - 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Вероятность", percent(probability))
    c2.metric("Справедливый кэф", f"{fair_odds:.2f}")
    c3.metric("Расчётное EV", f"{ev:+.1f}%")

    st.warning(
        "Это упрощённая модель. Положительное EV не доказывает наличие "
        "реального преимущества и не гарантирует прибыль."
    )

st.subheader("4. Хронологический бэктест")
st.caption("Каждый матч оценивается только по матчам, сыгранным раньше.")

max_start = len(df) - 5
start_n = st.number_input(
    "Матчей в начальном обучающем окне",
    min_value=10,
    max_value=max_start,
    value=min(100, max_start),
    step=10,
)

if st.button("Запустить бэктест"):
    predictions, actual = [], []
    progress = st.progress(0)

    for idx in range(int(start_n), len(df)):
        past = df.iloc[:idx]
        row = df.iloc[idx]
        r, ah, aa = fit_ratings(past)
        pred = predict(row.HomeTeam, row.AwayTeam, r, ah, aa)

        predictions.append([
            pred["Победа хозяев"],
            pred["Ничья"],
            pred["Победа гостей"],
        ])

        if row.HomeGoals > row.AwayGoals:
            actual.append(0)
        elif row.HomeGoals == row.AwayGoals:
            actual.append(1)
        else:
            actual.append(2)

        progress.progress((idx - int(start_n) + 1) / (len(df) - int(start_n)))

    probs = np.asarray(predictions)
    actual = np.asarray(actual)
    one_hot = np.eye(3)[actual]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    loss = float(log_loss(actual, probs, labels=[0, 1, 2]))
    accuracy = float(np.mean(probs.argmax(axis=1) == actual))

    st.subheader("Результаты бэктеста")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", percent(accuracy))
    c2.metric("Log loss", f"{loss:.3f}")
    c3.metric("Brier score", f"{brier:.3f}")
    st.caption(f"Проверено матчей: {len(actual)}.")
    st.warning(
        "Этот бэктест оценивает вероятности, а не доходность ставок. "
        "Модель упрощённая и требует сравнения с базовыми прогнозами."
    )

else:
st.info("Загрузи CSV с историей матчей, чтобы получить прогноз.")

st.divider()
st.caption(
"Автоматические матчи и коэффициенты пока не подключены. "
"Прогнозы вероятностные и не гарантируют финансового результата."
               )
