import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Заголовок и описание
st.title("INC 5000 Companies Analysis Dashboard")
st.write("Цей додаток дозволяє аналізувати дані про 5000 найшвидше зростаючих приватних компаній США за 2014 рік. \
Використовуйте панель фільтрів для налаштування перегляду.")

# Завантаження даних
@st.cache_data
def load_data():
    df = pd.read_csv("Data Set- Inc5000 Company List_2014.csv")
    df = df.drop(columns=["_input", "_num", "_widgetName", "_source", "_resultNumber", "_pageUrl", "id", "url"])
    df = df.dropna()
    return df

df = load_data()

# Фільтри
industry = st.selectbox("Оберіть індустрію", sorted(df["industry"].unique()))
states = st.multiselect("Оберіть штати", sorted(df["state_l"].unique()), default=["California", "Texas"])
revenue_filter = st.radio("Діапазон виручки", ["Всі", "Менше 10М", "Від 10М до 100М", "Більше 100М"])
growth_checkbox = st.checkbox("Показати лише компанії з ростом > 500%")

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

# Вибір колонок
columns_to_show = st.multiselect("Оберіть колонки для перегляду", filtered_df.columns,
                                 default=["company", "city", "state_l", "revenue", "growth"])
st.dataframe(filtered_df[columns_to_show])

# Графіки
st.subheader("Графічне представлення даних")

# 1: Гістограма росту
fig1, ax1 = plt.subplots()
sns.histplot(filtered_df["growth"], bins=30, ax=ax1)
ax1.set_title("Розподіл росту компаній")
st.pyplot(fig1)

# 2: Boxplot виручки по штатам
fig2, ax2 = plt.subplots()
sns.boxplot(x="state_l", y="revenue", data=filtered_df, ax=ax2)
ax2.set_title("Виручка по штатам")
ax2.tick_params(axis='x', rotation=90)
st.pyplot(fig2)

# 3: Scatterplot росту проти виручки
fig3, ax3 = plt.subplots()
sns.scatterplot(x="growth", y="revenue", hue="industry", data=filtered_df, ax=ax3)
ax3.set_title("Залежність росту і виручки по індустріях")
st.pyplot(fig3)

# Регресійна модель
st.subheader("Лінійна регресія: прогноз виручки за ростом")

X = filtered_df[["growth"]]
y = filtered_df["revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)
st.write(f"R² score моделі: {score:.2f}")

# Візуалізація регресії
fig4, ax4 = plt.subplots()
ax4.scatter(X_test, y_test, label="Фактичні значення")
ax4.plot(X_test, y_pred, color="red", label="Прогноз")
ax4.set_xlabel("Growth")
ax4.set_ylabel("Revenue")
ax4.set_title("Прогноз виручки по росту")
ax4.legend()
st.pyplot(fig4)
