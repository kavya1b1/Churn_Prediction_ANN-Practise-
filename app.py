import streamlit as st
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle

model=tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)


with open('label_encoder_gender.pkl','rb')as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

## Streamlit App

st.title('Customer Churn Prediction ML model')

## User Input
geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.number_input('Tenure',0,10)
num_of_products=st.number_input('Number Of Products')
has_cr_card=st.selectbox('Has Cr Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

## prepare the input data
input_data= pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})
## One-Hot-Encoding

geo_encode=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encode_df=pd.DataFrame(geo_encode,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

##conbining the One hot encoded column with the input data

input_data=pd.concat([input_data.reset_index(drop=True),geo_encode_df],axis=1)

input_data_scaled=scaler.transform(input_data)

## Prediction Churn 
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

st.write(f'Churn Probability :-{prediction_prob:.2f}')


if(prediction_prob>=0.5):
    st.write("The person will likely to churn")
else:
    st.write("The person will likely NOT to churn")





# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import streamlit as st
# import pandas as pd
# import tensorflow as tf
# import numpy as np
# import pickle

# # ---------------- LOAD RESOURCES SAFELY ----------------

# @st.cache_resource
# def load_resources():
#     model = tf.keras.models.load_model("model.h5")

#     with open("onehot_encoder_geo.pkl", "rb") as f:
#         onehot_encoder_geo = pickle.load(f)

#     with open("label_encoder_gender.pkl", "rb") as f:
#         label_encoder_gender = pickle.load(f)

#     with open("scaler.pkl", "rb") as f:
#         scaler = pickle.load(f)

#     return model, onehot_encoder_geo, label_encoder_gender, scaler

# model, onehot_encoder_geo, label_encoder_gender, scaler = load_resources()

# # ---------------- STREAMLIT UI ----------------

# st.title("Customer Churn Prediction ML Model")

# geography = st.selectbox(
#     "Geography",
#     onehot_encoder_geo.categories_[0]
# )

# gender = st.selectbox(
#     "Gender",
#     label_encoder_gender.classes_
# )

# age = st.slider("Age", 18, 92)
# balance = st.number_input("Balance")
# credit_score = st.number_input("Credit Score")
# estimated_salary = st.number_input("Estimated Salary")
# tenure = st.number_input("Tenure", 0, 10)
# num_of_products = st.number_input("Number Of Products", 1, 4)
# has_cr_card = st.selectbox("Has Credit Card", [0, 1])
# is_active_member = st.selectbox("Is Active Member", [0, 1])

# # ---------------- PREPARE INPUT ----------------

# input_data = pd.DataFrame({
#     "CreditScore": [credit_score],
#     "Gender": [label_encoder_gender.transform([gender])[0]],
#     "Age": [age],
#     "Tenure": [tenure],
#     "Balance": [balance],
#     "NumOfProducts": [num_of_products],
#     "HasCrCard": [has_cr_card],
#     "IsActiveMember": [is_active_member],
#     "EstimatedSalary": [estimated_salary]
# })

# # One-hot encode geography
# geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
# geo_df = pd.DataFrame(
#     geo_encoded,
#     columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
# )

# # Combine
# input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# # Align feature order (CRITICAL)
# input_data = input_data[scaler.feature_names_in_]

# # Scale
# input_scaled = scaler.transform(input_data)

# # Predict
# prediction = model.predict(input_scaled)
# prediction_prob = float(prediction[0][0])

# # ---------------- OUTPUT ----------------

# if prediction_prob >= 0.5:
#     st.error(f"Customer is likely to churn (Probability: {prediction_prob:.2f})")
# else:
#     st.success(f"Customer is NOT likely to churn (Probability: {prediction_prob:.2f})")
