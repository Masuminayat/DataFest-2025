from turtle import title
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.utils
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ==========================================
#  FORECASTING MODELS & DATA (From Code B)
# ==========================================

# === Load Models ===
# Note: Ensure these .pkl files are present in your directory
try:
    income_model = joblib.load("income_lr_model.pkl")
    poverty_model = joblib.load("poverty_lr_model.pkl")
    inflation_model = joblib.load("inflation_arimax_model.pkl")
except Exception as e:
    print(f"Model Loading Warning: {e}")
    # Fallback to prevent crash if files missing during testing
    income_model = None
    poverty_model = None
    inflation_model = None

# === Load historical data (Annual.xlsx) ===
try:
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

except Exception as e:
    print(f"Annual Data Load Error: {e}")
    df = pd.DataFrame()
    trend_values = None

# === Helper functions for Forecasting (From Code B) ===
def forecast_income(last_vals):
    """Predict Income using income_model"""
    if not income_model: return 0
    inc_input = np.array([[last_vals[c] for c in features_income]])
    return income_model.predict(inc_input)[0]

def forecast_inflation(last_vals):
    """Predict Inflation using inflation_model (ARIMAX)"""
    if not inflation_model: return 0
    exog_input = pd.DataFrame([{col: last_vals[col] for col in exog_vars}])
    return inflation_model.predict(n_periods=1, exogenous=exog_input).iloc[0]

def forecast_poverty(last_vals):
    """Predict Poverty using poverty_model"""
    if not poverty_model: return 0
    pov_input = np.array([[last_vals[c] for c in features_poverty]])
    return poverty_model.predict(pov_input)[0]


# ==========================================
#  OTHER EXCEL FILES (Original Code A Logic)
# ==========================================
dataframes = {}
files_map = {
    'Employment': ['Employment.xlsx', 'Employement.xlsx'],
    'Education': ['Education.xlsx'],
    'Household': ['Household.xlsx', 'House Hold.xlsx'],
    'Economy': ['Economy.xlsx'],
    'Health': ['Health.xlsx']
}

print("--- LOADING CHART DATA ---")
for key, file_list in files_map.items():
    for filename in file_list:
        if os.path.exists(filename):
            try:
                dataframes[key] = pd.read_excel(filename)
                print(f"✅ Loaded: {filename}")
                break
            except: pass
    if key not in dataframes:
        dataframes[key] = pd.DataFrame()
        print(f"❌ Missing: {key}")

# ==========================================
#  HELPER FUNCTION (Original Code A)
# ==========================================
def clean_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# ==========================================
#  API ENDPOINTS (Original Code A Charts)
# ==========================================

# 1. WORKFORCE SHIFT (Employment)
@app.get("/api/workforce-chart")
async def get_workforce_chart():
    if dataframes['Employment'].empty: return {"error": "File Missing"}
    try:
        df_chart = dataframes['Employment'].copy()
        indices = [1, 2, 5, 6] if "unnamed" in str(df_chart.columns[0]).lower() else [0, 1, 4, 5]
        
        df_chart = df_chart.iloc[:, indices].copy()
        df_chart.columns = ['Year', 'Area', 'Ag', 'NonAg']
        
        mask = df_chart['Area'].astype(str).str.strip().str.lower().str.contains('pakistan|total', regex=True, na=False)
        df_chart = df_chart[mask].copy()
        
        df_chart = clean_numeric(df_chart, ['Ag', 'NonAg'])
        df_chart.dropna(subset=['Ag', 'NonAg'], inplace=True)
        
        df_chart['SortYear'] = df_chart['Year'].astype(str).str[:4].astype(int)
        df_chart.sort_values('SortYear', inplace=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_chart['SortYear'].tolist(), y=df_chart['Ag'].tolist(), name='Agriculture', fill='tozeroy', line=dict(color='Orange')))
        fig.add_trace(go.Scatter(x=df_chart['SortYear'].tolist(), y=df_chart['NonAg'].tolist(), name='Non-Ag', fill='tozeroy', line=dict(color='Green')))
        fig.update_layout(title='Workforce Shift', yaxis_title='Percentage (%)', xaxis_title='Year', height=400, margin=dict(l=30, r=10, t=40, b=30))
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}

# 2. LITERACY RATE (Education)
@app.get("/api/literacy-chart")
async def get_literacy_chart():
    if dataframes['Education'].empty: return {"error": "File Missing"}
    try:
        df_chart = dataframes['Education'].copy()
        df_chart = df_chart.iloc[:, [0, 1, 13]].copy()
        df_chart.columns = ['Year', 'Region', 'Rate']
        
        mask = df_chart['Region'].astype(str).str.strip().str.lower().str.contains('overall|pakistan', regex=True, na=False)
        df_chart = df_chart[mask].copy()
        
        df_chart = clean_numeric(df_chart, ['Rate'])
        df_chart.dropna(subset=['Rate'], inplace=True)
        
        df_chart['SortYear'] = df_chart['Year'].astype(str).str[:4].astype(int)
        df_chart.sort_values('SortYear', inplace=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_chart['SortYear'].tolist(), y=df_chart['Rate'].tolist(), mode='lines+markers', name='Literacy', line=dict(color='Orange', width=3)))
        fig.update_layout(title='National Literacy Rate', yaxis_title='Percentage (%)', xaxis_title='Year', height=400, margin=dict(l=30, r=10, t=40, b=30))
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}

# 3. TOILET ACCESS (Household)
@app.get("/api/toilet-chart")
async def get_toilet_chart():
    if dataframes['Household'].empty: return {"error": "File Missing"}
    try:
        df_chart = dataframes['Household'].copy()
        indices = [6, 7, 10]
        df_chart = df_chart.iloc[:, indices].copy()
        df_chart.columns = ['Year', 'Source', 'Val']
        
        mask = df_chart['Source'].astype(str).str.strip().str.upper() == 'NO-TOILET'
        df_chart = df_chart[mask].copy()
        
        df_chart = clean_numeric(df_chart, ['Val'])
        val = df_chart['Val'].mean()
        if pd.isna(val): val = 0
        
        fig = go.Figure(go.Indicator(mode="gauge+number", value=val, title={'text': "No Toilet Access %"}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':'Green'}}))
        fig.update_layout(title='No Toilet Access %', height=400, margin=dict(l=30, r=30, t=50, b=30))
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}

# 4. INCOME vs CONSUMPTION (Economy)
@app.get("/api/economy-chart")
async def get_economy_chart():
    if dataframes['Economy'].empty: return {"error": "File Missing"}
    try:
        df_chart = dataframes['Economy'].copy()
        inc_col = next((c for c in df_chart.columns if "Income" in str(c)), df_chart.columns[2])
        cons_col = next((c for c in df_chart.columns if "Consumption" in str(c) or "Expenditure" in str(c)), df_chart.columns[3])
        
        df_chart = df_chart[[df_chart.columns[0], inc_col, cons_col]].copy()
        df_chart.columns = ['Year', 'Inc', 'Cons']
        
        df_chart = clean_numeric(df_chart, ['Inc', 'Cons'])
        df_chart = df_chart.groupby('Year').mean(numeric_only=True).reset_index()
        
        df_chart['SortYear'] = df_chart['Year'].astype(str).str[:4].astype(int)
        df_chart.sort_values('SortYear', inplace=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_chart['SortYear'].tolist(), y=df_chart['Inc'].tolist(), name='Income', line=dict(color='Green')))
        fig.add_trace(go.Scatter(x=df_chart['SortYear'].tolist(), y=df_chart['Cons'].tolist(), name='Consumption', line=dict(color='Orange')))
        fig.update_layout(title='Income vs Consumption', yaxis_title='Amount (PKR)', xaxis_title='Year', height=400, margin=dict(l=30, r=10, t=40, b=30))
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}

# 5. GENDER PARITY (Education)
@app.get("/api/gpi-chart")
async def get_gpi_chart():
    if dataframes['Education'].empty: return {"error": "File Missing"}
    try:
        df_chart = dataframes['Education'].copy()
        df_chart = df_chart.iloc[:, [0, 1, 11, 12]].copy()
        df_chart.columns = ['Year', 'Region', 'Male', 'Female']
        
        mask = df_chart['Region'].astype(str).str.strip().str.lower().str.contains('overall|pakistan', regex=True, na=False)
        df_chart = df_chart[mask].copy()
        
        df_chart = clean_numeric(df_chart, ['Male', 'Female'])
        df_chart['GPI'] = df_chart['Female'] / df_chart['Male']
        df_chart.dropna(subset=['GPI'], inplace=True)
        
        df_chart['SortYear'] = df_chart['Year'].astype(str).str[:4].astype(int)
        df_chart.sort_values('SortYear', inplace=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_chart['SortYear'].tolist(), y=df_chart['GPI'].tolist(), mode='lines+markers', name='GPI', line=dict(color='Green', width=3)))
        fig.update_layout(title='Literacy GPI (Female/Male)', yaxis_title='GPI', xaxis_title='Year', height=400, margin=dict(l=30, r=10, t=40, b=30))
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}

# 6. IMMUNIZATION (Health)
@app.get("/api/immunization-chart")
async def get_immunization_chart():
    if dataframes['Health'].empty: return {"error": "File Missing"}
    try:
        df_chart = dataframes['Health'].copy()
        df_chart = df_chart.iloc[:, [0, 1, 10]].copy()
        df_chart.columns = ['Year', 'Region', 'Val']
        
        latest_year = df_chart['Year'].dropna().unique()[-1]
        df_chart = df_chart[df_chart['Year'] == latest_year].copy()
        
        provs = ['punjab', 'sindh', 'kpk', 'nwfp', 'balochistan']
        mask = df_chart['Region'].astype(str).str.strip().str.lower().isin(provs)
        df_chart = df_chart[mask].copy()
        
        df_chart = clean_numeric(df_chart, ['Val'])
        df_chart.sort_values('Val', inplace=True)
        
        fig = go.Figure(go.Bar(x=df_chart['Val'].tolist(), y=df_chart['Region'].tolist(), orientation='h', marker=dict(color='Orange')))
        fig.update_layout(title=f'Immunization by Province ({latest_year})', yaxis_title='Province', xaxis_title='Percentage (%)', height=400, margin=dict(l=80, r=10, t=40, b=30))
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}

# 7. DIARRHOEA (Health)
@app.get("/api/diarrhoea-chart")
async def get_diarrhoea_chart():
    if dataframes['Health'].empty: return {"error": "File Missing"}
    try:
        df_chart = dataframes['Health'].copy()
        d_col = next((c for c in df_chart.columns if "DIARRHOEA" in str(c).upper()), None)
        if not d_col: return {"error": "Col Not Found"}
        
        df_chart = df_chart[[df_chart.columns[0], df_chart.columns[1], d_col]].copy()
        df_chart.columns = ['Year', 'Region', 'Val']
        
        mask = df_chart['Region'].astype(str).str.lower().str.contains('overall|pakistan', regex=True, na=False)
        df_chart = df_chart[mask].copy()
        
        df_chart = clean_numeric(df_chart, ['Val'])
        df_chart['SortYear'] = df_chart['Year'].astype(str).str[:4].astype(int)
        df_chart.sort_values('SortYear', inplace=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_chart['SortYear'].tolist(), y=df_chart['Val'].tolist(), mode='lines+markers', line=dict(color='Green')))
        fig.update_layout(title='Diarrhoea Incidence', yaxis_title='Percentage (%)', xaxis_title='Year', height=400, margin=dict(l=30, r=10, t=40, b=30))
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}

# 8. TETANUS (Health)
@app.get("/api/tetanus-chart")
async def get_tetanus_chart():
    if dataframes['Health'].empty: return {"error": "File Missing"}
    try:
        df_chart = dataframes['Health'].copy()
        t_col = next((c for c in df_chart.columns if "TETANUS" in str(c).upper()), None)
        if not t_col: return {"error": "Col Not Found"}
        
        df_chart = df_chart[[df_chart.columns[0], df_chart.columns[1], t_col]].copy()
        df_chart.columns = ['Year', 'Region', 'Val']
        
        mask = df_chart['Region'].astype(str).str.lower().str.contains('overall|pakistan', regex=True, na=False)
        df_chart = df_chart[mask].copy()
        
        df_chart = clean_numeric(df_chart, ['Val'])
        df_chart['SortYear'] = df_chart['Year'].astype(str).str[:4].astype(int)
        df_chart.sort_values('SortYear', inplace=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_chart['SortYear'].tolist(), y=df_chart['Val'].tolist(), mode='lines+markers', line=dict(color='orange')))
        fig.update_layout(title='Tetanus Coverage', yaxis_title='Percentage (%)', xaxis_title='Year', height=400, margin=dict(l=30, r=10, t=40, b=30))
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}

# 9. TENURE (Household)
@app.get("/api/tenure-chart")
async def get_tenure_chart():
    if dataframes['Household'].empty: return {"error": "File Missing"}
    try:
        df_chart = dataframes['Household'].copy()
        indices = [12, 13, 14, 15]
        df_chart = df_chart.iloc[:, indices].copy()
        df_chart.columns = ['Year', 'Region', 'Owner', 'Rent']
        
        mask = df_chart['Region'].astype(str).str.lower().str.contains('overall|pakistan', regex=True, na=False)
        df_chart = df_chart[mask].copy()
        latest = df_chart.iloc[-1]
        
        vals = [pd.to_numeric(latest['Owner'], errors='coerce'), pd.to_numeric(latest['Rent'], errors='coerce')]
        fig = go.Figure(data=[go.Pie(labels=['Owner', 'Rent'], values=vals, hole=.4, marker=dict(colors=['Green', 'Orange']))])
        fig.update_layout(title='House Tenure', yaxis_title='Percentage (%)', xaxis_title='Tenure Type', height=400, margin=dict(l=10, r=10, t=40, b=10))
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}

# 10. DISPARITY (Household)
@app.get("/api/disparity-chart")
async def get_disparity_chart():
    if dataframes['Household'].empty: return {"error": "File Missing"}
    try:
        df_chart = dataframes['Household'].copy()
        categories = ['Electricity', 'Flush Toilet', 'Tap Water']
        urban_vals = []
        rural_vals = []
        
        # --- 1. ELECTRICITY ---
        try:
            df_elec = df_chart.iloc[:, [20, 21, 22]].copy()
            df_elec.columns = ['Year', 'Region', 'Percentage']
            latest_year_elec = df_elec['Year'].dropna().unique()[-1]
            subset_elec = df_elec[df_elec['Year'] == latest_year_elec].copy()
            subset_elec['Percentage'] = pd.to_numeric(subset_elec['Percentage'], errors='coerce')
            u_elec = subset_elec[subset_elec['Region'].astype(str).str.strip().str.lower() == 'urban']['Percentage'].max()
            r_elec = subset_elec[subset_elec['Region'].astype(str).str.strip().str.lower() == 'rural']['Percentage'].max()
            urban_vals.append(0 if pd.isna(u_elec) else u_elec)
            rural_vals.append(0 if pd.isna(r_elec) else r_elec)
        except: urban_vals.append(0); rural_vals.append(0)

        # --- 2. FLUSH TOILET ---
        try:
            df_toilet = df_chart.iloc[:, [6, 7, 8, 9]].copy()
            df_toilet.columns = ['Year', 'Source', 'Urban', 'Rural']
            mask_flush = df_toilet['Source'].astype(str).str.strip().str.upper() == 'FLUSH'
            df_flush = df_toilet[mask_flush].copy()
            latest_year_t = df_flush['Year'].dropna().unique()[-1]
            last_row = df_flush[df_flush['Year'] == latest_year_t].iloc[-1]
            u_t = pd.to_numeric(last_row['Urban'], errors='coerce')
            r_t = pd.to_numeric(last_row['Rural'], errors='coerce')
            urban_vals.append(0 if pd.isna(u_t) else u_t)
            rural_vals.append(0 if pd.isna(r_t) else r_t)
        except: urban_vals.append(0); rural_vals.append(0)

        # --- 3. TAP WATER ---
        try:
            df_water = df_chart.iloc[:, [0, 1, 2, 3]].copy()
            df_water.columns = ['Year', 'Source', 'Urban', 'Rural']
            mask_water = df_water['Source'].astype(str).str.strip().str.upper().str.contains('PIPED', na=False)
            df_pipe = df_water[mask_water].copy()
            latest_year_w = df_pipe['Year'].dropna().unique()[-1]
            last_row_w = df_pipe[df_pipe['Year'] == latest_year_w].iloc[-1]
            u_w = pd.to_numeric(last_row_w['Urban'], errors='coerce')
            r_w = pd.to_numeric(last_row_w['Rural'], errors='coerce')
            urban_vals.append(0 if pd.isna(u_w) else u_w)
            rural_vals.append(0 if pd.isna(r_w) else r_w)
        except: urban_vals.append(0); rural_vals.append(0)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=categories, y=urban_vals, name='Urban', marker_color='Green', text=[f'{x:.0f}%' for x in urban_vals], textposition='auto'))
        fig.add_trace(go.Bar(x=categories, y=rural_vals, name='Rural', marker_color='Orange', text=[f'{x:.0f}%' for x in rural_vals], textposition='auto'))
        fig.update_layout(title='Urban vs. Rural Basic Access Disparity (Latest)', yaxis_title='Percentage (%)', xaxis_title='Category', barmode='group', height=400, margin=dict(l=30, r=10, t=40, b=30))
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}

# ==========================================
#  MISSING CHARTS (ADDED NOW)
# ==========================================

# 11. NATIONAL SAVINGS RATE TREND (Economy)
@app.get("/api/savings-chart")
async def get_savings_chart():
    if dataframes['Economy'].empty: return {"error": "File Missing"}
    try:
        df_chart = dataframes['Economy'].copy()
        # Find relevant columns dynamically
        inc_col = next((c for c in df_chart.columns if "Income" in str(c)), df_chart.columns[2])
        sav_col = next((c for c in df_chart.columns if "Saving" in str(c)), df_chart.columns[4])
        
        df_chart = df_chart[[df_chart.columns[0], inc_col, sav_col]].copy()
        df_chart.columns = ['Year', 'Income', 'Savings']
        
        df_chart = clean_numeric(df_chart, ['Income', 'Savings'])
        
        # Calculate Rate
        df_chart['Savings_Rate'] = (df_chart['Savings'] / df_chart['Income']) * 100
        
        # Group by Year (National Avg)
        df_chart = df_chart.groupby('Year')['Savings_Rate'].mean().reset_index()
        
        df_chart['SortYear'] = df_chart['Year'].astype(str).str[:4].astype(int)
        df_chart.sort_values('SortYear', inplace=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_chart['SortYear'].tolist(), 
            y=df_chart['Savings_Rate'].tolist(), 
            mode='lines+markers', 
            name='Savings Rate %', 
            line=dict(color='DarkGreen', width=3),
            fill='tozeroy', # Area chart effect
            fillcolor='rgba(0, 100, 0, 0.1)'
        ))
        
        fig.update_layout(
            title='<b>National Savings Rate Trend</b>', 
            yaxis_title='Savings % of Income',
            xaxis_title='Year',
            height=400, 
            margin=dict(l=30, r=10, t=40, b=30),
            hovermode="x unified"
        )
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}


# 12. VULNERABLE EMPLOYMENT - Urban vs Rural (Employment)
@app.get("/api/vulnerable-emp-chart")
async def get_vulnerable_chart():
    if dataframes['Employment'].empty: return {"error": "File Missing"}
    try:
        df_chart = dataframes['Employment'].copy()
        # Indices based on your Notebook inspection
        # Year(0), Area(1), Ag(4), Unpaid(6)
        indices = [0, 1, 4, 6] 
        df_chart = df_chart.iloc[:, indices].copy()
        df_chart.columns = ['Year', 'Area', 'Agriculture', 'Unpaid']
        
        df_chart = clean_numeric(df_chart, ['Agriculture', 'Unpaid'])
        df_chart['Vulnerable'] = df_chart['Agriculture'] + df_chart['Unpaid']
        
        # Filter Urban/Rural only
        mask = df_chart['Area'].astype(str).str.strip().str.lower().isin(['urban', 'rural'])
        df_chart = df_chart[mask].copy()
        
        # Sort
        df_chart['SortYear'] = df_chart['Year'].astype(str).str[:4].astype(int)
        df_chart.sort_values(['SortYear', 'Area'], inplace=True)
        
        fig = go.Figure()
        
        # Add Urban Trace
        urban_data = df_chart[df_chart['Area'].str.lower() == 'urban']
        fig.add_trace(go.Bar(
            x=urban_data['SortYear'].tolist(),
            y=urban_data['Vulnerable'].tolist(),
            name='Urban Vulnerable',
            marker_color='Orange'
        ))
        
        # Add Rural Trace
        rural_data = df_chart[df_chart['Area'].str.lower() == 'rural']
        fig.add_trace(go.Bar(
            x=rural_data['SortYear'].tolist(),
            y=rural_data['Vulnerable'].tolist(),
            name='Rural Vulnerable',
            marker_color='Green'
        ))

        fig.update_layout(
            title='<b>Vulnerable Employment (Ag + Unpaid)</b>',
            yaxis_title='Number of Vulnerable Workers',
            xaxis_title='Year',
            barmode='group',
            height=400, 
            margin=dict(l=30, r=10, t=40, b=30)
        )
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}


# 13. CORE INFRASTRUCTURE DONUT (Household)
@app.get("/api/infrastructure-donut")
async def get_infra_chart():
    if dataframes['Household'].empty: return {"error": "File Missing"}
    try:
        df = dataframes['Household'].copy()
        
        # Helper to get latest National Average for a specific label
        def get_metric(year_idx, label_idx, val_idx, label_target):
            temp = df.iloc[:, [year_idx, label_idx, val_idx]].copy()
            temp.columns = ['Year', 'Label', 'Value']
            temp['Value'] = pd.to_numeric(temp['Value'], errors='coerce')
            
            # Filter for specific label (e.g., FLUSH) and latest year
            mask = temp['Label'].astype(str).str.contains(label_target, case=False, na=False)
            latest_year = temp['Year'].dropna().unique()[-1]
            val = temp[(temp['Year'] == latest_year) & mask]['Value'].mean()
            return val if not pd.isna(val) else 0

        # Based on Notebook columns:
        elec_val = get_metric(20, 21, 22, 'TOTAL') # Electricity
        toilet_val = get_metric(6, 7, 10, 'FLUSH') # Flush Toilet
        water_val = get_metric(0, 1, 3, 'PIPED')   # Tap Water

        labels = ['Electricity', 'Flush Toilet', 'Tap Water']
        values = [elec_val, toilet_val, water_val]
        colors = ['Green', 'Orange', '#03A9F4']

        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=.5, 
            marker=dict(colors=colors),
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title='<b>Core Infrastructure Access (Latest)</b>',
            height=400, 
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=True

        )
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e: return {"error": str(e)}


# ==========================================
#  PREDICTION ENDPOINT (From Code B)
# ==========================================
@app.post("/predict_future")
async def predict_future(target: str = Form(...), years_ahead: int = Form(...)):
    target = target.strip()
    # Note: Code B assertion handles validation, wrapping in try/except for API safety
    try:
        assert target in ['Poverty', 'Avg_Income_filled', 'Inflation'], \
            "Target must be Poverty, Avg_Income_filled, or Inflation"

        if df.empty: return JSONResponse({"error": "Annual Data not loaded"})

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
                # Chain: Income -> Inflation -> Poverty
                inc_pred = forecast_income(last_vals)
                last_vals['Avg_Income_filled'] = inc_pred

                infl_pred = forecast_inflation(last_vals)
                last_vals['Inflation'] = infl_pred

                pov_pred = forecast_poverty(last_vals)
                last_vals['Poverty'] = pov_pred

            elif target == 'Avg_Income_filled':
                # Chain: Poverty -> Inflation -> Income
                pov_pred = forecast_poverty(last_vals)
                last_vals['Poverty'] = pov_pred

                infl_pred = forecast_inflation(last_vals)
                last_vals['Inflation'] = infl_pred

                inc_pred = forecast_income(last_vals)
                last_vals['Avg_Income_filled'] = inc_pred

            else:  # target == 'Inflation'
                # Chain: Income -> Poverty -> Inflation
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
    except Exception as e:
        return JSONResponse({"error": str(e)})