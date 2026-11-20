"""Synthetic (SIMULATED) data for a Tamil Nadu green hydrogen plant."""
import numpy as np
import pandas as pd

HOURS = np.arange(24)

def weather_day(rng, cloud=None, wind_mean=None):
    """One synthetic day of weather: cloud cover (0-1) and wind speed (m/s)."""
    cloud = rng.uniform(0.0, 0.7) if cloud is None else cloud
    wind_mean = rng.uniform(3.5, 8.5) if wind_mean is None else wind_mean
    cl = np.clip(cloud + rng.normal(0, 0.08, 24), 0, 1)
    # TN wind: stronger in afternoon/evening and at night (monsoon-style pattern)
    diurnal = 1 + 0.25 * np.sin((HOURS - 10) / 24 * 2 * np.pi)
    ws = np.clip(wind_mean * diurnal + rng.normal(0, 0.8, 24), 0, 25)
    return cl, ws

def solar_cf(cl):
    """Physics-style solar capacity factor from cloud cover (ground truth)."""
    sun = np.clip(np.sin((HOURS - 6) / 12 * np.pi), 0, None)
    return sun * (1 - 0.75 * cl ** 1.5)

def wind_cf(ws):
    """Simple turbine power curve: cut-in 3, rated 12, cut-out 25 m/s."""
    cf = np.clip((ws - 3) / (12 - 3), 0, 1) ** 3
    cf[ws >= 25] = 0
    return cf

def make_training_set(days=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        cl, ws = weather_day(rng)
        sc, wc = solar_cf(cl), wind_cf(ws)
        for h in HOURS:
            rows.append(dict(day=d, hour=h, cloud=cl[h], wind=ws[h],
                             solar_cf=sc[h] + rng.normal(0, 0.02),
                             wind_cf=wc[h] + rng.normal(0, 0.03)))
    return pd.DataFrame(rows)

# Illustrative time-of-day grid tariff (Rs/kWh) - SIMULATED, not an official TANGEDCO tariff
GRID_PRICE = np.array([5.0]*5 + [6.5]*1 + [8.5]*4 + [7.0]*8 + [9.5]*4 + [8.0]*1 + [5.0]*1)[:24]
# Illustrative grid emission factor (kg CO2/kWh): lower at midday (solar), higher at night/evening
GRID_EF = np.array([0.80]*6 + [0.74]*3 + [0.62]*3 + [0.55]*4 + [0.70]*2 + [0.85]*4 + [0.82]*2)[:24]
RENEWABLE_PRICE = 3.6   # Rs/kWh, assumed PPA / captive renewable cost
RENEWABLE_EF = 0.0      # kg CO2/kWh (operational)