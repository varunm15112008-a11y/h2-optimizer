import numpy as np
import pandas as pd
import streamlit as st
from data import GRID_EF, GRID_PRICE, RENEWABLE_PRICE, weather_day
from forecast import predict_day, train_models
from optimizer import baseline_24x7, pareto, solve

st.set_page_config(page_title="H2 Optimizer", layout="wide")


@st.cache_resource
def get_models():
    return train_models()


models, mae = get_models()

# ---------------- Sidebar: what-if controls ----------------
st.sidebar.title("H2 Optimizer")
st.sidebar.caption("All plant and grid data are SIMULATED.")
cloud = st.sidebar.slider("Cloud cover (0 clear - 1 overcast)", 0.0, 1.0, 0.25, 0.05)
wind = st.sidebar.slider("Average wind speed (m/s)", 2.0, 12.0, 6.5, 0.5)
solar_mw = st.sidebar.slider("Solar plant (MWp)", 2, 30, 12)
wind_mw = st.sidebar.slider("Wind farm (MW)", 0, 15, 5)
cap_mw = st.sidebar.slider("Electrolyser size (MW)", 2, 20, 10)
demand = st.sidebar.slider("Hydrogen demand (kg/day)", 500, 4000, 2500, 100)
balance = st.sidebar.slider(
    "Priority: CO2 <---> Cost",
    0.0,
    1.0,
    0.5,
    0.1,
    help="0 = minimise CO2 only, 1 = minimise cost only, 0.5 = balanced",
)
kwh_kg = st.sidebar.slider("Electrolyser energy (kWh per kg H2)", 48, 60, 52)
min_load = st.sidebar.slider("Minimum load (% of capacity)", 10, 40, 20) / 100
seed = st.sidebar.number_input("Weather noise seed", 0, 999, 5)

# ---------------- Compute ----------------
rng = np.random.default_rng(int(seed))
cl, ws = weather_day(rng, cloud=cloud, wind_mean=wind)
s_cf, w_cf = predict_day(models, cl, ws)
solar_kwh, wind_kwh = s_cf * solar_mw * 1000, w_cf * wind_mw * 1000
gen = solar_kwh + wind_kwh
cap_kw = cap_mw * 1000
kw = dict(cap_kw=cap_kw, kwh_per_kg=kwh_kg, min_load=min_load)
base = baseline_24x7(gen, demand, kwh_per_kg=kwh_kg)
front, _ = pareto(gen, demand, **kw)
res = solve(gen, demand, w=balance, cost_ref=base["cost"], co2_ref=base["co2"], **kw)

st.title("H2 Optimizer - green hydrogen plant scheduler")
st.caption("Tells a plant WHEN to run its electrolysers so hydrogen costs less AND emits less CO2.")

if res is None:
    st.error("Demand cannot be met with this electrolyser size. Increase the electrolyser size or lower the demand.")
    st.stop()

hours = [f"{h:02d}:00" for h in range(24)]
tabs = st.tabs(["Overview", "Forecast", "Schedule", "Cost vs CO2", "Plant health", "Report", "About / assumptions"])

# ---- Overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Cost per kg H2", f"Rs {res['cost_per_kg']:.0f}", f"{(res['cost']/base['cost']-1)*100:.0f}% vs 24/7")
    c[1].metric("Rupees saved today", f"Rs {base['cost']-res['cost']:,.0f}")
    c[2].metric("CO2 avoided today", f"{base['co2']-res['co2']:,.0f} kg", f"{(res['co2']/base['co2']-1)*100:.0f}% vs 24/7")
    c[3].metric("Hydrogen produced", f"{res['kg']:,.0f} kg")
    st.write("**How it works:** forecast renewable output (ML) -> optimise the hourly load (MILP) -> compare with running flat 24/7.")
    st.line_chart(
        pd.DataFrame(
            {"Electrolyser load (MW)": res["P"] / 1000, "Renewable available (MW)": gen / 1000},
            index=hours,
        )
    )

# ---- Forecast
with tabs[1]:
    st.subheader("AI forecast of renewable output (next 24 h)")
    st.line_chart(pd.DataFrame({"Solar (MW)": solar_kwh / 1000, "Wind (MW)": wind_kwh / 1000}, index=hours))
    st.write(
        f"Model: Gradient Boosting (scikit-learn). Held-out test error (MAE): "
        f"solar {mae['solar_cf']*100:.1f}% of capacity, wind {mae['wind_cf']*100:.1f}% of capacity."
    )
    st.write(f"Total renewable energy forecast: **{gen.sum()/1000:.1f} MWh**")

# ---- Schedule
with tabs[2]:
    st.subheader("Optimised hourly schedule")
    df = pd.DataFrame({"Renewable used (MW)": res["R"] / 1000, "Grid used (MW)": res["G"] / 1000}, index=hours)
    st.bar_chart(df)
    st.line_chart(pd.DataFrame({"Grid price (Rs/kWh)": GRID_PRICE, "Grid CO2 (x10 kg/kWh)": GRID_EF * 10}, index=hours))
    st.subheader("Why this schedule?")
    load = res["P"] / cap_kw
    on = [h for h in range(24) if load[h] > 0.01]
    peak = [h for h in range(24) if load[h] > 0.9]
    ren_share = res["R"].sum() / res["P"].sum() * 100
    base_ren_share = base["R"].sum() / max(base["P"].sum(), 1e-9) * 100
    st.write(f"- Electrolysers run in **{len(on)} of 24 hours**, at full load in **{len(peak)} hours**.")
    st.write(f"- **{ren_share:.0f}%** of the electricity comes from renewables (24/7 baseline: {base_ren_share:.0f}%).")
    if peak:
        st.write(
            f"- Highest load at {hours[peak[0]]}-{hours[peak[-1]]} when renewable output is high and grid power is not needed."
        )
    st.write("- Grid power is used mainly in hours where it is cheap or low-carbon, depending on the priority slider.")
    st.dataframe(
        pd.DataFrame(
            {
                "Hour": hours,
                "Load (%)": (load * 100).round(0),
                "Renewable (MWh)": (res["R"] / 1000).round(2),
                "Grid (MWh)": (res["G"] / 1000).round(2),
            }
        ),
        hide_index=True,
    )

# ---- Cost vs CO2
with tabs[3]:
    st.subheader("The cost vs CO2 trade-off")
    fr = pd.DataFrame(
        {
            "Priority (0=CO2, 1=cost)": [w for w, _ in front],
            "Cost (Rs)": [r["cost"] for _, r in front],
            "CO2 (kg)": [r["co2"] for _, r in front],
        }
    )
    st.scatter_chart(fr, x="Cost (Rs)", y="CO2 (kg)")
    st.dataframe(fr.round(0), hide_index=True)
    st.info(
        "Each point is the best possible schedule for one priority setting. Moving along the curve trades cost for CO2. "
        "Renewable power helps BOTH, so the trade-off only appears for the part of the demand that must come from the grid. "
        "Use the sidebar slider to choose your balance (0.5 = balanced)."
    )

# ---- Plant health
with tabs[4]:
    st.subheader("Plant health (simulated sensor readings)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Water purification**")
        cond = st.slider("Feed-water conductivity (uS/cm)", 0.0, 5.0, 0.6, 0.1)
        tank = st.slider("Purified water tank level (%)", 0, 100, 70)
        need_l = res["kg"] * 9 * 1.3  # ~9 L water per kg H2 + purification losses
        st.write(f"Water needed today: about **{need_l:,.0f} L**")
        if cond > 1.0:
            st.error("Startup BLOCKED: water too conductive (limit 1 uS/cm, assumption).")
        elif tank < 25:
            st.warning("Startup DELAYED: tank level low.")
        else:
            st.success("Water ready - electrolysers can start.")
    with c2:
        st.markdown("**Machine condition**")
        temp = st.slider("Stack temperature (C)", 40, 100, 70)
        eff = st.slider("Stack efficiency (% of rated)", 60, 100, 92)
        if temp > 85:
            st.error("Stack too hot: reduce load and inspect cooling.")
        elif eff < 80:
            st.warning(f"Efficiency down to {eff}%: energy per kg is higher, schedule maintenance.")
        else:
            st.success("Machine healthy.")
        if eff < 100:
            st.caption(f"At {eff}% efficiency, real energy use is about {kwh_kg/(eff/100):.1f} kWh per kg.")

# ---- Report
with tabs[5]:
    st.subheader("Daily manufacturer report")
    rep = pd.DataFrame(
        {
            "Hour": hours,
            "Load_MW": (res["P"] / 1000).round(3),
            "Renewable_MWh": (res["R"] / 1000).round(3),
            "Grid_MWh": (res["G"] / 1000).round(3),
            "Grid_price_Rs_kWh": GRID_PRICE,
            "Grid_CO2_kg_kWh": GRID_EF,
        }
    )
    st.write(
        f"Hydrogen: **{res['kg']:,.0f} kg** | Cost: **Rs {res['cost']:,.0f}** | "
        f"CO2: **{res['co2']:,.0f} kg** | Saved vs 24/7: **Rs {base['cost']-res['cost']:,.0f}** cost, "
        f"**{base['co2']-res['co2']:,.0f} kg** CO2."
    )
    st.dataframe(rep, hide_index=True)
    st.download_button("Download report (CSV)", rep.to_csv(index=False), "h2_daily_report.csv", "text/csv")

# ---- About
with tabs[6]:
    st.subheader("About and assumptions")
    st.markdown(
        f"""
**All data here are simulated** for demonstration; no real plant or grid data are used.
- Electrolyser: {cap_mw} MW, {kwh_kg} kWh/kg H2, minimum load {min_load*100:.0f}%, ramp limit 50% of capacity per hour.
- Renewable cost Rs {RENEWABLE_PRICE}/kWh (assumed), zero operational CO2.
- Grid tariff and CO2 factor vary by hour (illustrative, not official values).
- Baseline: electrolysers run at a flat load all 24 hours.
- Forecast model trained on synthetic Tamil Nadu-style weather. Next step: real weather API and plant data.
- Method: mixed-integer linear programming minimising w x Cost + (1-w) x CO2 (both normalised).
"""
    )