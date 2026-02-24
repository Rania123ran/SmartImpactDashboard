import pandas as pd
import numpy as np

# Generate 2 years of daily data (730 days)
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=730, freq='D')

# Base metrics with seasonality
day_of_year = dates.dayofyear
seasonality = np.sin(day_of_year * (2 * np.pi / 365))

revenue = np.random.normal(loc=3000, scale=500, size=730) + (seasonality * 500)
margin = revenue * np.random.uniform(0.15, 0.25, size=730)
energy_consumption = np.random.normal(loc=150, scale=20, size=730) + (np.abs(seasonality) * 30)
co2_emissions = energy_consumption * 0.23
absenteeism_rate = np.random.normal(loc=3.0, scale=0.8, size=730)
csat = np.random.normal(loc=85, scale=4, size=730)

df = pd.DataFrame({
    'Date': dates,
    'Revenue_MAD': np.round(revenue, 2),
    'Margin_MAD': np.round(margin, 2),
    'Energy_kWh': np.round(energy_consumption, 2),
    'CO2_Emissions_kg': np.round(co2_emissions, 2),
    'Absenteeism_Pct': np.round(absenteeism_rate, 2),
    'Customer_Satisfaction': np.round(csat, 2)
})

# --- INJECT ANOMALIES ---
# 1. Equipment failure causing an energy spike over a few days
df.loc[200:205, 'Energy_kWh'] *= 1.8
df.loc[200:205, 'CO2_Emissions_kg'] = df.loc[200:205, 'Energy_kWh'] * 0.23

# 2. HR issue causing a prolonged absenteeism spike and CSAT drop
df.loc[500:510, 'Absenteeism_Pct'] = np.random.normal(12.0, 1.5, 11)
df.loc[500:510, 'Customer_Satisfaction'] = np.random.normal(65.0, 5.0, 11)

# Save to CSV
df.to_csv('simulated_kpi_data_large.csv', index=False)