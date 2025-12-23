import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

# page configuration

st.set_page_config("Linear Regression App",layout="centered")


#Load css

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

load_css("style.css")

#Title

st.markdown("""<div class="card">
    <h1> Linear Regression </h1>
    <p> Predict <b> Tip Amount </b> from <b> Total Bill </b> using Linear Regression..</p>
    </div>  """,unsafe_allow_html=True)

#Load Data 
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()

#Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())
#st.markdown('<div class="card">',unsafe_allow_html=True)
#st.subheader("Dataset Preview")
#st.dataframe(df.head())
#st.markdown('</div>',unsafe_allow_html=True)

#prepare data
x,y=df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

#Train Model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#metrics
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mae)
r2=r2_score(y_test,y_pred)
adjusted_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)

#Visualization
#st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Visualization")
fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"],alpha=0.6)
ax.plot(df["total_bill"],model.predict(scaler.transform(df[["total_bill"]])),color="red")
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
st.pyplot(fig)
#st.markdown('</div>',unsafe_allow_html=True)

#performance 

#st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Model Performance")
c1,c2=st.columns(2)
c1.metric("Mean Absolute Error (MAE)",f"{mae:.2f}")
c2.metric ("Root Mean Squared Error (RMSE)",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("R Squared",f"{r2:.3f}")
c4.metric("Adjusted R Squared",f"{adjusted_r2:.3f}")
#st.markdown('</div>',unsafe_allow_html=True)

#  m&c
st.markdown(f"""<div class="card-footer">
            <h3>Model Intercept & Coefficient</h3>
            <p><b>Coefficient:</b> {model.coef_[0]:.3f} </p>
            <p><b>Intercept:</b> {model.intercept_:.3f} </p>
            </div>  """,unsafe_allow_html=True)

#Prediction

#st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Predict Tip Amount")
bill=st.slider("Total Bill",float(df.total_bill.min()),float(df.total_bill.max()),30.0)
tip=model.predict(scaler.transform([[bill]]))[0]
st.markdown(f'<div class="Prediction-box">Predicted Tip :${tip:.2f}</div>',unsafe_allow_html=True)
#st.markdown('</div>',unsafe_allow_html=True)