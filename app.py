
import streamlit as st
import pickle


def load_files():
    # import model files, label dictionary and scaler
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('label_dics.pkl', 'rb') as f:
        label_dict = pickle.load(f)

    with open('scale.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return model, label_dict, scaler


st.header("Banking Marketing Campaign Prediction System")
st.subheader("Enter the following details to predict the likelihood of customers subscription:")

model, label_dics, scaler = load_files()

# create form
default = st.selectbox("default", ['no', 'unknown', 'yes'])
default = label_dics['default'][default]
contact = st.selectbox("contact", ['cellular', 'telephone'])
contact = label_dics['contact'][contact]
month = st.selectbox("month", ['apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep'])
month = label_dics['month'][month]
day_of_week = st.selectbox("day_of_week", ['fri', 'mon', 'thu', 'tue', 'wed'])
day_of_week = label_dics['day_of_week'][day_of_week]
pdays = st.number_input("pdays", min_value=0)
poutcome = st.selectbox("poutcome", ['failure', 'nonexistent', 'success'])
poutcome = label_dics['poutcome'][poutcome]
emp_var_rate = st.number_input("emp.var.rate")
cons_conf_idx = st.number_input("cons.conf.idx")
euribor3m = st.number_input("euribor3m")
nr_employed = st.number_input("nr.employed", min_value=1)


input_data = [default, contact, month, day_of_week, pdays, poutcome, emp_var_rate, cons_conf_idx, euribor3m, nr_employed]


# predict using model and display result
if st.button("Predict"):
    input_data = scaler.transform([input_data])
    prediction = model.predict_proba(input_data)

    st.write("The likelihood of customer subscribing: ", str(prediction[0][1]*100)+'%')

