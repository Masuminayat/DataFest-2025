# National Socio-Economic Intelligence System (ML & BI)
### 🏆 Featured Project at DataFest 2026 | Pakistan Bureau of Statistics

An end-to-end Predictive Intelligence System designed to analyze 20 years of Pakistan's socio-economic historical data. This project integrates a **FastAPI** backend with **Machine Learning (ARIMAX & Regression)** and **Power BI** to forecast national trends in poverty, income, and infrastructure access.

---

## 🚀 Key Features
- **Predictive Analytics:** Forecasting Inflation, Poverty, and Income trends using ARIMAX and Multi-variate Linear Regression.
- **Dynamic Simulation:** A "Feature Prediction" engine allowing users to manipulate GDP, Literacy, and Unemployment variables to observe real-time impacts on poverty.
- **Geospatial & Equity Insights:** Regional drill-downs identifying urban-rural gaps in sanitation, health (immunization), and education.
- **Interactive Visuals:** High-fidelity dashboards built in Power BI and real-time forecasting charts using Plotly.

---

## 📊 Socio-Economic Outcomes & Findings
Based on the analysis of **20 years of historical government data**, the system uncovered:
- **Income Forecast:** Successfully predicted a national average income trajectory reaching **22K**.
- **Sanitation Gap:** Quantified a **30%+ Urban-Rural divide** in basic sanitation infrastructure.
- **Housing vs. Utility:** Discovered that while **85.3%** of the population are homeowners, only **61.4%** have access to flush toilets.
- **Education & Gender:** Identified a **13.7% Literacy Gender Gap** and a **51.2% Total Literacy Rate** across provinces.
- **Health Vulnerability:** Mapped a **67.3% Full Immunization Rate**, pinpointing regions at risk for disease outbreaks.

---

## 🛠️ Technical Stack
- **Backend:** FastAPI (Python)
- **Machine Learning:** Scikit-Learn (Linear Regression), Pmdarima (ARIMAX for Time-Series)
- **Data Processing:** Pandas, NumPy
- **Frontend:** HTML5, CSS3 (AOS Animation Library), Plotly.js
- **Business Intelligence:** Power BI (DAX, Data Modeling)
- **Deployment:** Joblib (Model Serialization)

---

## 🏗️ System Architecture
1. **Data Ingestion:** Processed two decades of annual socio-economic indicators from the Pakistan Bureau of Statistics.
2. **Feature Engineering:** Calculated Year-over-Year (YoY) growth, urban-to-rural ratios, and gender parity indices.
3. **Modeling:** - `inflation_arimax_model.pkl`: Handles exogenous economic shocks.
    - `poverty_lr_model.pkl`: Correlates GDP/Literacy with poverty levels.
4. **API Layer:** FastAPI serves predictions via POST requests, feeding the dynamic frontend.

---
