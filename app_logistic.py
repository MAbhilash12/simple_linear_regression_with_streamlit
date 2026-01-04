import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# ---------------------------------------------------
# PAGE CONFIG (FIRST STREAMLIT COMMAND)
# ---------------------------------------------------
st.set_page_config(
    page_title="Logistic Regression App",
    page_icon="📊",
    layout="centered"
)

# ---------------------------------------------------
# LOAD CSS (AFTER set_page_config)
# ---------------------------------------------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style4.css")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.markdown("""
<div class="card">
<h1>Logistic Regression</h1>
<p>Binary Classification using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())


# ---------------------------------------------------
# Features & Target
# ---------------------------------------------------
X = df.drop("target", axis=1)
y = df["target"]

# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------------------------------------------------
# Feature Scaling
# ---------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# Train Logistic Regression
# ---------------------------------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------
# Predictions
# ---------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"**Accuracy:** {accuracy:.2f}")

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------
st.subheader("🔍 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig_cm)

# ---------------------------------------------------
# Classification Report
# ---------------------------------------------------
st.subheader("📄 Classification Report")
st.text(classification_report(y_test, y_pred))

# ---------------------------------------------------
# ROC Curve
# ---------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

st.subheader("📊 ROC Curve")

fig_roc, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig_roc)

# ---------------------------------------------------
# User Input Prediction
# ---------------------------------------------------
st.subheader("🧪 Make a Prediction")

user_input = []
for feature in X.columns[:5]:  # Using first 5 features for simplicity
    value = st.slider(
        feature,
        float(X[feature].min()),
        float(X[feature].max()),
        float(X[feature].mean())
    )
    user_input.append(value)

# Pad remaining features with mean values
remaining_features = list(X.mean()[5:])
final_input = np.array(user_input + remaining_features).reshape(1, -1)

final_input_scaled = scaler.transform(final_input)
prediction = model.predict(final_input_scaled)[0]
prediction_prob = model.predict_proba(final_input_scaled)[0][1]

st.markdown("---")
st.subheader("✅ Prediction Result")

if prediction == 1:
    st.success(f"Benign (Probability: {prediction_prob:.2f})")
else:
    st.error(f"Malignant (Probability: {prediction_prob:.2f})")
