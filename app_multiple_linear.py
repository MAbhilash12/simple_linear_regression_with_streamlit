import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="Multiple Linear Regression App", layout="centered")

# --------------------------------------------------
# Load CSS
# --------------------------------------------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style2.css")

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown("""
<div class="card">
    <h1>Multiple Linear Regression</h1>
    <p>Predict <b>Tip Amount</b> using <b>Total Bill</b> and <b>Party Size</b></p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Prepare Data (MLR)
# --------------------------------------------------
X = df[["total_bill", "size"]]
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)   # ✅ FIXED (no fit_transform)

# --------------------------------------------------
# Train Model
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --------------------------------------------------
# Metrics (Corrected)
# --------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # ✅ FIXED
r2 = r2_score(y_test, y_pred)

adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.subheader("Actual vs Predicted Tip")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)
ax.set_xlabel("Actual Tip")
ax.set_ylabel("Predicted Tip")
st.pyplot(fig)

# --------------------------------------------------
# Model Performance
# --------------------------------------------------
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R² Score", f"{r2:.3f}")
c4.metric("Adjusted R²", f"{adjusted_r2:.3f}")

# --------------------------------------------------
# Model Coefficients
# --------------------------------------------------
st.markdown(f"""
<div class="card-footer">
    <h3>Model Parameters</h3>
    <p><b>Total Bill Coefficient:</b> {model.coef_[0]:.3f}</p>
    <p><b>Size Coefficient:</b> {model.coef_[1]:.3f}</p>
    <p><b>Intercept:</b> {model.intercept_:.3f}</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill ($)",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

size = st.slider(
    "Party Size",
    int(df['size'].min()),
    int(df['size'].max()),
    2
)

input_data = scaler.transform([[bill, size]])
predicted_tip = model.predict(input_data)[0]

st.markdown(
    f'<div class="Prediction-box">Predicted Tip: ${predicted_tip:.2f}</div>',
    unsafe_allow_html=True
)
