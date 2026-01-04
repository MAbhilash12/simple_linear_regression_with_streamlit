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

# ---------------- CLEAR CACHE (FIRST RUN FIX) ----------------
st.cache_data.clear()

# ---------------- TITLE ----------------
st.title("Multiple Linear Regression")
st.write("California Housing Price Prediction")

# ---------------- IMAGE ----------------
st.image(
    "https://images.unsplash.com/photo-1570129477492-45c003edd2be",
    use_container_width=True
)

# ---------------- LOAD DATA (LOCAL CSV ONLY) ----------------
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

feature_columns = X.columns.tolist()

# ---------------- HANDLE MISSING VALUES ----------------
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=feature_columns
)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- MODEL TRAINING ----------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# ---------------- METRICS ----------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_train.shape[1] - 1)

# ---------------- PERFORMANCE ----------------
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:,.2f}")
c2.metric("RMSE", f"{rmse:,.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adjusted R²", f"{adjusted_r2:.3f}")

# ---------------- COEFFICIENTS ----------------
st.subheader("Feature Coefficients")

coef_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

st.dataframe(coef_df)

# ===================== VISUALIZATIONS =====================

# Actual vs Predicted
st.subheader("Actual vs Predicted Values")
fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, alpha=0.6)
ax1.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")
ax1.set_xlabel("Actual Values")
ax1.set_ylabel("Predicted Values")
st.pyplot(fig1)

# Residuals Plot
st.subheader("Residuals vs Predicted")
residuals = y_test - y_pred
fig2, ax2 = plt.subplots()
ax2.scatter(y_pred, residuals, alpha=0.6)
ax2.axhline(0)
ax2.set_xlabel("Predicted Values")
ax2.set_ylabel("Residuals")
st.pyplot(fig2)

# Feature Importance
st.subheader("Feature Importance")
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.barh(coef_df["Feature"], coef_df["Coefficient"])
ax3.set_xlabel("Coefficient Value")
st.pyplot(fig3)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

# ---------------- PREDICTION ----------------
st.subheader("Predict House Price")

input_data = {}
for col in feature_columns:
    input_data[col] = st.number_input(
        label=col,
        value=float(df[col].median())
    )

input_df = pd.DataFrame([input_data])
input_df = input_df[feature_columns]

input_df = pd.DataFrame(
    imputer.transform(input_df),
    columns=feature_columns
)

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

st.success(f"Predicted House Value: ${prediction * 100000:,.0f}")
