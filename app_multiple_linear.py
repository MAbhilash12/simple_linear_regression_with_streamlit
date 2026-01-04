import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multiple Linear Regression App",
    layout="centered"
)

# ---------------- LOAD CSS ----------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style2.css")

# ---------------- TITLE ----------------
st.markdown("""
<div class="card">
    <h1>Multiple Linear Regression</h1>
    <p>Predict <b>California House Prices</b> using Multiple Linear Regression</p>
</div>
""", unsafe_allow_html=True)

# ---------------- IMAGE ----------------
st.image(
    "https://images.unsplash.com/photo-1570129477492-45c003edd2be",
    caption="California Housing Price Prediction",
    use_container_width=True
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")

df = load_data()

# ---------------- DATA PREVIEW ----------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- ONE-HOT ENCODING ----------------
df = pd.get_dummies(
    df,
    columns=["ocean_proximity"],
    drop_first=True
)

# ---------------- FEATURES & TARGET ----------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# 🔑 SAVE FEATURE ORDER
feature_columns = X.columns

# ---------------- IMPUTE MISSING VALUES ----------------
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=feature_columns
)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- TRAIN MODEL ----------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# ---------------- METRICS ----------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_train.shape[1] - 1)

# ---------------- PERFORMANCE ----------------
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:,.2f}")
c2.metric("RMSE", f"{rmse:,.2f}")

c3, c4 = st.columns(2)
c3.metric("R Squared", f"{r2:.3f}")
c4.metric("Adjusted R Squared", f"{adjusted_r2:.3f}")

# ---------------- COEFFICIENTS ----------------
st.subheader("Feature Coefficients")

coef_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

st.dataframe(coef_df)

# ===================== GRAPHS =====================
st.subheader("Model Visualizations")

# 1️⃣ Actual vs Predicted
st.markdown("### Actual vs Predicted Values")
fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, alpha=0.6)
ax1.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
ax1.set_xlabel("Actual House Value")
ax1.set_ylabel("Predicted House Value")
st.pyplot(fig1)

# 2️⃣ Residuals Plot
st.markdown("### Residuals vs Predicted")
residuals = y_test - y_pred
fig2, ax2 = plt.subplots()
ax2.scatter(y_pred, residuals, alpha=0.6)
ax2.axhline(0)
ax2.set_xlabel("Predicted Values")
ax2.set_ylabel("Residuals")
st.pyplot(fig2)

# 3️⃣ Feature Importance
st.markdown("### Feature Importance (Coefficients)")
fig3, ax3 = plt.subplots()
ax3.barh(coef_df["Feature"], coef_df["Coefficient"])
ax3.set_xlabel("Coefficient Value")
ax3.set_ylabel("Feature")
st.pyplot(fig3)

# 4️⃣ Correlation Heatmap
st.markdown("### Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(9, 7))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

# ---------------- PREDICTION ----------------
st.subheader("Predict House Price")

input_data = {}
for col in feature_columns:
    input_data[col] = st.slider(
        col,
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )

input_df = pd.DataFrame([input_data])

# 🔑 ENSURE SAME COLUMN ORDER
input_df = input_df.reindex(columns=feature_columns)

input_imputed = pd.DataFrame(
    imputer.transform(input_df),
    columns=feature_columns
)

input_scaled = scaler.transform(input_imputed)

prediction = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted House Value: ${prediction * 100000:,.0f}</div>',
    unsafe_allow_html=True
)
