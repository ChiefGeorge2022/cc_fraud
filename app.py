import streamlit as st
import pickle
import pandas as pd


#---------------Page Setup-----------------#

# st.set_page_config(layout="wide")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Set Title
st.title("Credit Card Fraud Detection")
st.image('credit_card.png')
st.markdown('#')

# --------------- Slider Section-----------------------------
st.sidebar.markdown(f" ## :gear: Credit Card Features")
st.sidebar.write("Each sliders are used to adjust the credit card's transaction details.")
st.sidebar.markdown('---')


# Fuction for loading and caching the model
@st.cache(allow_output_mutation=True)
def load_data():
    model = pickle.load(open("XGB_model.pkl", "rb"))
    return model

#Load xgb model
XG_model = load_data()


#----------------Dataset Section---------------#
data = pd.read_csv('Important_Features.csv')

def features():
    V14 = st.sidebar.slider('V14', data.V14.min(), data.V14.max(), float(data.V14.median()))
    V10 = st.sidebar.slider('V10', data.V10.min(), data.V10.max(), float(data.V10.median()))
    V8 = st.sidebar.slider('V8', data.V8.min(), data.V8.max(), float(data.V8.median()))
    V25 = st.sidebar.slider('V25', data.V25.min(), data.V25.max(), float(data.V25.median()))
    V12 = st.sidebar.slider('V12', data.V12.min(), data.V12.max(), float(data.V12.median()))
    V4 = st.sidebar.slider('V4', data.V4.min(), data.V4.max(), float(data.V4.median()))
    V17 = st.sidebar.slider('V17', data.V17.min(), data.V17.max(), float(data.V17.median()))
    V27 = st.sidebar.slider('V27', data.V27.min(), data.V27.max(), float(data.V27.median()))
    V5 = st.sidebar.slider('V5', data.V5.min(), data.V5.max(), float(data.V5.median()))
    V19 = st.sidebar.slider('V19', data.V19.min(), data.V19.max(), float(data.V19.median()))
    
    input_data = {'V14':V14,'V10':V10, 'V8':V8,'V25':V25,'V12':V12,'V4':V4,'V17':V17,'V27':V27,'V5':V5,'V19':V19}

    # Create input data
    input_features = pd.DataFrame(input_data, index=[0])
    
    return input_features

# collect user inputs
df = features()


# -----------------Predictions-----------------------

predictions = XG_model.predict(df)

st.markdown(f"### Model Prediction")

if predictions == 0:
    st.success('This credit card transaction is valid!')
else:
    st.error('This credit card transaction is Fraudulent!')

