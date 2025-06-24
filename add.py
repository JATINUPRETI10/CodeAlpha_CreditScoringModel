import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# Load dataset
url = "https://raw.githubusercontent.com/jwu424/GiveMeSomeCredit/master/cs-training.csv"
df = pd.read_csv(url, index_col=0)
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0], inplace=True)

X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

def predict_credit_risk(
    RevolvingUtilizationOfUnsecuredLines, age,
    NumberOfTime30_59DaysPastDueNotWorse, DebtRatio,
    MonthlyIncome, NumberOfOpenCreditLinesAndLoans,
    NumberOfTimes90DaysLate, NumberRealEstateLoansOrLines,
    NumberOfTime60_89DaysPastDueNotWorse, NumberOfDependents
):
    input_data = [
        RevolvingUtilizationOfUnsecuredLines, age,
        NumberOfTime30_59DaysPastDueNotWorse, DebtRatio,
        MonthlyIncome, NumberOfOpenCreditLinesAndLoans,
        NumberOfTimes90DaysLate, NumberRealEstateLoansOrLines,
        NumberOfTime60_89DaysPastDueNotWorse, NumberOfDependents
    ]
    input_scaled = scaler.transform([input_data])
    pred = rf_model.predict(input_scaled)[0]
    prob = rf_model.predict_proba(input_scaled)[0][1]
    result = "❌ Not Creditworthy" if pred == 1 else "✅ Creditworthy"
    return f"{result} (Risk Score: {prob:.2f})"

inputs = [
    gr.Number(label="RevolvingUtilizationOfUnsecuredLines"),
    gr.Number(label="Age"),
    gr.Number(label="NumberOfTime30-59DaysPastDueNotWorse"),
    gr.Number(label="DebtRatio"),
    gr.Number(label="MonthlyIncome"),
    gr.Number(label="NumberOfOpenCreditLinesAndLoans"),
    gr.Number(label="NumberOfTimes90DaysLate"),
    gr.Number(label="NumberRealEstateLoansOrLines"),
    gr.Number(label="NumberOfTime60-89DaysPastDueNotWorse"),
    gr.Number(label="NumberOfDependents"),
]

app = gr.Interface(
    fn=predict_credit_risk,
    inputs=inputs,
    outputs=gr.Text(label="Prediction"),
    title="🔍 Credit Scoring Predictor",
    description="Enter user data to check if they are creditworthy",
)

app.launch()

