import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import altair as alt


# Заголовок и описание
st.set_page_config(
    page_title="INC 5000 Companies Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': 'https://github.com/streamlit/streamlit/issues',
        'About': 'Цей додаток дозволяє аналізувати дані про 5000 найшвидше зростаючих приватних компаній США за 2014 рік. \
Використовуйте панель фільтрів для налаштування перегляду.'
    }
)

# Завантаження даних
@st.cache_data
def load_data():
    df = pd.read_csv("Data Set- Inc5000 Company List_2014.csv")
    df = df.drop(columns=["_input", "_num", "_widgetName", "_source", "_resultNumber", "_pageUrl", "id", "url"])
    df = df.dropna()
    return df

df = load_data()

# Фільтри
st.sidebar.title("Панель фільтрації")

industry = st.sidebar.selectbox("Оберіть індустрію", sorted(df["industry"].unique()))
states = st.sidebar.multiselect("Оберіть штати", sorted(df["state_l"].unique()), default=["California", "Texas"])
revenue_filter = st.sidebar.radio("Діапазон виручки", ["Всі", "Менше 10М", "Від 10М до 100М", "Більше 100М"])
growth_checkbox = st.sidebar.checkbox("Показати лише компанії з ростом > 500%")
chart_option = st.sidebar.radio(
    "📈 Оберіть графік для перегляду:",
    [
        "Дохід проти зростання",
        "Гістограма росту",
        "Boxplot виручки по штатам",
        "Scatterplot росту проти виручки"
    ]
)

# Інформаційний блок
st.sidebar.markdown("---")
st.sidebar.markdown(" **Інструкція**: \nФільтруйте дані за параметрами і переглядайте графіки та таблиці на панелі праворуч.")
st.sidebar.markdown(" **Автор**: Parkhomuk Amina")

# Застосування фільтрів
filtered_df = df[df["industry"] == industry]
if states:
    filtered_df = filtered_df[filtered_df["state_l"].isin(states)]
if revenue_filter != "Всі":
    if revenue_filter == "Менше 10М":
        filtered_df = filtered_df[filtered_df["revenue"] < 1e7]
    elif revenue_filter == "Від 10М до 100М":
        filtered_df = filtered_df[(filtered_df["revenue"] >= 1e7) & (filtered_df["revenue"] <= 1e8)]
    else:
        filtered_df = filtered_df[filtered_df["revenue"] > 1e8]
if growth_checkbox:
    filtered_df = filtered_df[filtered_df["growth"] > 500]

st.title("📊 INC 5000 Companies Analysis Dashboard")

# Вибір колонок
st.subheader("Оберіть, які стовпці таблиці відображати")
columns_to_show = st.multiselect("Оберіть колонки для перегляду", filtered_df.columns,
                                 default=["company", "city", "state_l", "revenue", "growth"])
st.dataframe(filtered_df[columns_to_show])

# Графіки

# 1: Дохід проти зростання
if chart_option == "Дохід проти зростання":
    st.subheader("📊 Дохід проти зростання")
    chart = alt.Chart(filtered_df).mark_circle(size=60).encode(
        x=alt.X('growth:Q', title='Зростання (%)'),
        y=alt.Y('revenue:Q', title='Дохід (USD)'),
        color=alt.Color('industry:N', title='Індустрія'),
        tooltip=['company:N', 'growth:Q', 'revenue:Q', 'industry:N']
    ).interactive().properties(
        title="Залежність доходу від зростання"
    )
    st.altair_chart(chart, use_container_width=True)

# 2: Гістограма росту
if chart_option == " Гістограма росту":
    st.subheader("📊 Розподіл росту компаній")
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_df["growth"], bins=30, ax=ax1)
    st.pyplot(fig1)

# 3: Boxplot виручки по штатам
if chart_option == "Boxplot виручки по штатам":
    st.subheader("📊 Виручка по штатам")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="state_l", y="revenue", data=filtered_df, ax=ax2)
    ax2.tick_params(axis='x', rotation=90)
    st.pyplot(fig2)

# 4: Scatterplot росту проти виручки
if chart_option == "Scatterplot росту проти виручки":
    st.subheader("📊 Залежність росту і виручки по індустріях")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x="growth", y="revenue", hue="industry", data=filtered_df, ax=ax3)
    st.pyplot(fig3)

# --- Блок побудови регресії за вибором користувача ---
st.sidebar.markdown("### Побудова регресії")
numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

reg_x = st.sidebar.selectbox("Оберіть змінну X (незалежна)", numeric_columns, index=0)
reg_y = st.sidebar.selectbox("Оберіть змінну Y (залежна)", numeric_columns, index=1)
show_regression = st.sidebar.checkbox("Показати регресійну модель")

if show_regression:
    st.subheader(f"📈 Лінійна регресія: {reg_y} ~ {reg_x}")
    df_reg = df[[reg_x, reg_y]].dropna()

    if df_reg.shape[0] >= 2:
        model = LinearRegression()
        model.fit(df_reg[[reg_x]], df_reg[reg_y])
        y_pred = model.predict(df_reg[[reg_x]])

        coef = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(df_reg[[reg_x]], df_reg[reg_y])

        st.markdown(f"**Коефіцієнт нахилу (β):** {coef:.4f}")
        st.markdown(f"**Зсув (intercept):** {intercept:.4f}")
        st.markdown(f"**R²:** {r2:.4f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df_reg, x=reg_x, y=reg_y, ax=ax)
        sns.lineplot(x=df_reg[reg_x], y=y_pred, color='red', ax=ax)
        ax.set_title(f"Регресія {reg_y} ~ {reg_x}")
        st.pyplot(fig)
    else:
        st.warning("Недостатньо даних для побудови регресії.")
