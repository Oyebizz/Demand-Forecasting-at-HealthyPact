import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import gradio as gr
import os

class DemandForecasting:
    def __init__(self):
        self.raw_data = self.generate_synthetic_sales_data()
        self.preprocessed_data = None
        self.model = None

    def generate_synthetic_sales_data(self, n_periods=1000):
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=n_periods)
        product_categories = ['Organic Food', 'Supplements', 'Health Drinks']
        data = {
            'date': dates,
            'product_category': np.random.choice(product_categories, n_periods),
            'base_sales': np.random.normal(1000, 200, n_periods),
            'seasonality_factor': np.sin(np.linspace(0, 4 * np.pi, n_periods)) * 200 + 200,
            'economic_indicator': np.random.normal(100, 20, n_periods),
            'marketing_spend': np.random.normal(5000, 1000, n_periods),
        }
        df = pd.DataFrame(data)
        df['total_sales'] = (df['base_sales'] +
                             df['seasonality_factor'] +
                             df['economic_indicator'] / 10 +
                             df['marketing_spend'] / 100 +
                             np.random.normal(0, 100, n_periods))
        return df

    def preprocess_data(self):
        self.raw_data['month'] = self.raw_data['date'].dt.month
        self.raw_data['quarter'] = self.raw_data['date'].dt.quarter
        X = self.raw_data[['product_category', 'base_sales', 'seasonality_factor',
                           'economic_indicator', 'marketing_spend', 'month', 'quarter']]
        y = self.raw_data['total_sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), ['base_sales', 'seasonality_factor', 'economic_indicator', 'marketing_spend']),
            ('cat', OneHotEncoder(), ['product_category'])
        ])
        return X_train, X_test, y_train, y_test, preprocessor

    def train_model(self, X_train, y_train, preprocessor):
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        return pipeline


# Initialize the model
forecast = DemandForecasting()
X_train, X_test, y_train, y_test, preprocessor = forecast.preprocess_data()
pipeline = forecast.train_model(X_train, y_train, preprocessor)

# Define the prediction function
def predict_sales(product_category, base_sales, seasonality_factor, economic_indicator, marketing_spend, month, quarter):
    input_data = pd.DataFrame({
        "product_category": [product_category],
        "base_sales": [base_sales],
        "seasonality_factor": [seasonality_factor],
        "economic_indicator": [economic_indicator],
        "marketing_spend": [marketing_spend],
        "month": [month],
        "quarter": [quarter],
    })
    return pipeline.predict(input_data)[0]

# Create the Gradio interface
app = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Dropdown(["Organic Food", "Supplements", "Health Drinks"], label="Product Category"),
        gr.Number(label="Base Sales"),
        gr.Number(label="Seasonality Factor"),
        gr.Number(label="Economic Indicator"),
        gr.Number(label="Marketing Spend"),
        gr.Slider(1, 12, step=1, label="Month"),
        gr.Slider(1, 4, step=1, label="Quarter"),
    ],
    outputs=gr.Textbox(label="Predicted Sales"),
    title="HealthyPact Demand Forecasting"
)

# Launch the Gradio app
app.launch()
import gradio as gr

# Define the prediction function
def predict_sales(product_category, base_sales, seasonality_factor, economic_indicator, marketing_spend, month, quarter):
    return f"Predicted sales for {product_category} with base sales {base_sales}!"

# Define the Gradio Interface
app = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Dropdown(["Organic Food", "Supplements", "Health Drinks"], label="Product Category"),
        gr.Number(label="Base Sales"),
        gr.Number(label="Seasonality Factor"),
        gr.Number(label="Economic Indicator"),
        gr.Number(label="Marketing Spend"),
        gr.Slider(1, 12, step=1, label="Month"),
        gr.Slider(1, 4, step=1, label="Quarter"),
    ],
    outputs=gr.Textbox(label="Predicted Sales"),
    title="HealthyPact Demand Forecasting"
)

# Launch the Gradio app
app.launch()
import gradio as gr

# Define the prediction function
def predict_sales(product_category, base_sales, seasonality_factor, economic_indicator, marketing_spend, month, quarter):
    return f"Predicted sales for {product_category} with base sales {base_sales}!"

# Define the Gradio Interface
app = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Dropdown(["Organic Food", "Supplements", "Health Drinks"], label="Product Category"),
        gr.Number(label="Base Sales"),
        gr.Number(label="Seasonality Factor"),
        gr.Number(label="Economic Indicator"),
        gr.Number(label="Marketing Spend"),
        gr.Slider(1, 12, step=1, label="Month"),
        gr.Slider(1, 4, step=1, label="Quarter"),
    ],
    outputs=gr.Textbox(label="Predicted Sales"),
    title="HealthyPact Demand Forecasting"

# Launch the Gradio app
app.launch()



