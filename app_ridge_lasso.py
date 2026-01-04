import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config("Regression App", layout="centered")

# ---------------- LOAD CSS ----------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------------- TITLE ----------------
st.markdown("""
<div class="card">
    <h1>Regression Models</h1>
    <p>Predict <b>Tip Amount</b> from <b>Total Bill</b> using 
    <b>Linear, Ridge, and Lasso Regression</b>.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# ---------------- DATASET PREVIEW ----------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- PREPARE DATA ----------------
X = df[["total_bill"]]
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL SELECTION ----------------
st.subheader("Choose Regression Model")

model_type = st.selectbox(
    "Select Model",
    ["Linear Regression", "Ridge Regression", "Lasso Regression"]
)

alpha = 1.0
if model_type != "Linear Regression":
    alpha = st.slider("Regularization Strength (Alpha)", 0.01, 10.0, 1.0)

# ---------------- TRAIN MODEL ----------------
if model_type == "Linear Regression":
    model = LinearRegression()
elif model_type == "Ridge Regression":
    model = Ridge(alpha=alpha)
else:
    model = Lasso(alpha=alpha)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)

# ---------------- VISUALIZATION ----------------
st.subheader("Visualization")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)
ax.plot(
    df["total_bill"],
    model.predict(scaler.transform(df[["total_bill"]])),
    color="red"
)
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
ax.set_title(model_type)

st.pyplot(fig)

# ---------------- PERFORMANCE ----------------
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
c2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R Squared", f"{r2:.3f}")
c4.metric("Adjusted R Squared", f"{adjusted_r2:.3f}")

# ---------------- COEFFICIENTS ----------------
st.markdown(f"""
<div class="card-footer">
    <h3>Model Parameters</h3>
    <p><b>Model:</b> {model_type}</p>
    <p><b>Coefficient:</b> {model.coef_[0]:.3f}</p>
    <p><b>Intercept:</b> {model.intercept_:.3f}</p>
</div>
""", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

tip = model.predict(scaler.transform([[bill]]))[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip : ${tip:.2f}</div>',
    unsafe_allow_html=True
)
