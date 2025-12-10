from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === Load Models ===
income_model = joblib.load("income_lr_model.pkl")
poverty_model = joblib.load("poverty_lr_model.pkl")
inflation_model = joblib.load("inflation_arimax_model.pkl")  # ARIMAX

# === Load historical data ===
df = pd.read_excel("Annual.xlsx")
df.set_index("Year", inplace=True)
df = df[['GDP_per_Capita', 'Unemployment', 'Literacy_Rate', 'Poverty', 'Avg_Income_filled', 'Inflation']]

# Base columns used for linear trend extrapolation
base_cols = ['GDP_per_Capita', 'Unemployment', 'Literacy_Rate']

# Model features
features_income = ['GDP_per_Capita', 'Inflation', 'Unemployment', 'Literacy_Rate', 'Poverty']
features_poverty = ['GDP_per_Capita', 'Inflation', 'Unemployment', 'Literacy_Rate', 'Avg_Income_filled']
exog_vars = ['GDP_per_Capita', 'Poverty', 'Avg_Income_filled', 'Unemployment', 'Literacy_Rate']

# Compute last 3-year trends for base variables
trend_values = df[base_cols].tail(3).diff().mean()


# === Helper functions ===
def forecast_income(last_vals):
    """Predict Income using income_model"""
    inc_input = np.array([[last_vals[c] for c in features_income]])
    return income_model.predict(inc_input)[0]

def forecast_inflation(last_vals):
    """Predict Inflation using inflation_model (ARIMAX)"""
    exog_input = pd.DataFrame([{col: last_vals[col] for col in exog_vars}])
    return inflation_model.predict(n_periods=1, exogenous=exog_input).iloc[0]

def forecast_poverty(last_vals):
    """Predict Poverty using poverty_model"""
    pov_input = np.array([[last_vals[c] for c in features_poverty]])
    return poverty_model.predict(pov_input)[0]


# === FastAPI Routes ===
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_future")
async def predict_future(target: str = Form(...), years_ahead: int = Form(...)):
    target = target.strip()
    assert target in ['Poverty', 'Avg_Income_filled', 'Inflation'], \
        "Target must be Poverty, Avg_Income_filled, or Inflation"

    # Start from last known values
    last_vals = df.iloc[-1].to_dict()
    results = []

    # Determine starting year for forecast
    start_year = max(2024, int(df.index[-1]) + 1)
    last_year = start_year - 1

    for i in range(years_ahead):
        # Step 0: Update base columns by trend
        for col in base_cols:
            last_vals[col] += trend_values[col]

        # === Chained Forecasting ===
        if target == 'Poverty':
            # Chain: Income → Inflation → Poverty
            inc_pred = forecast_income(last_vals)
            last_vals['Avg_Income_filled'] = inc_pred

            infl_pred = forecast_inflation(last_vals)
            last_vals['Inflation'] = infl_pred

            pov_pred = forecast_poverty(last_vals)
            last_vals['Poverty'] = pov_pred

        elif target == 'Avg_Income_filled':
            # Chain: Poverty → Inflation → Income
            pov_pred = forecast_poverty(last_vals)
            last_vals['Poverty'] = pov_pred

            infl_pred = forecast_inflation(last_vals)
            last_vals['Inflation'] = infl_pred

            inc_pred = forecast_income(last_vals)
            last_vals['Avg_Income_filled'] = inc_pred

        else:  # target == 'Inflation'
            # Chain: Income → Poverty → Inflation
            inc_pred = forecast_income(last_vals)
            last_vals['Avg_Income_filled'] = inc_pred

            pov_pred = forecast_poverty(last_vals)
            last_vals['Poverty'] = pov_pred

            infl_pred = forecast_inflation(last_vals)
            last_vals['Inflation'] = infl_pred

        # Forecasted year
        forecast_year = last_year + i + 1

        # Append forecasted values with JSON-safe types
        results.append({
            "year": int(forecast_year),
            "Inflation": float(round(last_vals['Inflation'], 2)),
            "Avg_Income_filled": float(round(last_vals['Avg_Income_filled'], 2)),
            "Poverty": float(round(last_vals['Poverty'], 2)),
        })

    # Include last 10 years of historical data for comparison
    hist_years = df.index[-10:].tolist()
    actual_data = df.tail(10).to_dict(orient='records')

    # Convert historical data to JSON-safe types
    for row in actual_data:
        for key in row:
            if isinstance(row[key], (np.integer, np.int64)):
                row[key] = int(row[key])
            elif isinstance(row[key], (np.floating, np.float64)):
                row[key] = float(row[key])

    return JSONResponse({
        "predictions": results,
        "historical": actual_data,
        "hist_years": [int(y) for y in hist_years]
    })
