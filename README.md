Customer Churn Prediction using ANN (Streamlit App)

This project is a Customer Churn Prediction system built using an Artificial Neural Network (ANN) and deployed using Streamlit.
It predicts whether a customer is likely to churn (leave the bank) based on demographic and financial features.
------------------------------------------------------------------------------------------------------------------
🚀 Project Overview

Customer churn is a critical problem in the banking industry. Retaining existing customers is more cost-effective than acquiring new ones.

This project:

Trains an ANN model on customer data

Uses preprocessing techniques like Label Encoding, One-Hot Encoding, and Feature Scaling

Provides an interactive Streamlit web application for predictions
------------------------------------------------------------------------------------------------------------------

🧠 Machine Learning Pipeline

Data Preprocessing

Label Encoding (Gender)

One-Hot Encoding (Geography)

Feature Scaling using StandardScaler

Model Architecture

Input Layer

Hidden Layers with ReLU activation

Output Layer with Sigmoid activation

Loss Function

Binary Crossentropy

Optimizer

Adam

Evaluation Metric

Accuracy
------------------------------------------------------------------------------------------------------------------

📂 Project Structure
CHURN(ANN)/
├── app.py                       # Streamlit application

├── experiments.ipynb           # Model training & experimentation

├── prediction.ipynb            # Prediction testing notebook

├── requirements.txt            # Project dependencies

├── model.h5                    # Trained ANN model

├── model_tf215.h5              # TensorFlow compatible model

├── label_encoder_gender.pkl    # Gender label encoder

├── onehot_encoder_geo.pkl      # Geography one-hot encoder

├── scaler.pkl                  # Feature scaler

├── logs/                       # TensorBoard logs (ignored in GitHub)\

└── Churn_Modelling 2.csv       # Dataset (ignored in GitHub)
------------------------------------------------------------------------------------------------------------------

🖥️ Streamlit Web App Features

User-friendly input form

Real-time churn prediction

Probability-based decision

Lightweight and fast inference
------------------------------------------------------------------------------------------------------------------

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

2️⃣ Create and Activate Environment (Recommended)
conda create -n churn310 python=3.10 -y
conda activate churn310

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit App
streamlit run app.py

Open your browser at:

http://localhost:8501
------------------------------------------------------------------------------------------------------------------

📦 Dependencies

Key libraries used:

Python 3.10

TensorFlow / Keras

Scikit-learn

Pandas

NumPy

Streamlit

(Exact versions are listed in requirements.txt)
