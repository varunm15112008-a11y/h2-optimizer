import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from data import GRID_EF, GRID_PRICE, RENEWABLE_PRICE


def solve(
    gen_kwh,
    demand_kg,
    cap_kw=10000,
    kwh_per_kg=52.0,
    min_load=0.2,
    ramp=0.5,
    w=0.5,
    price=GRID_PRICE,
    ef=GRID_EF,
    ren_price=RENEWABLE_PRICE,
    cost_ref=None,
    co2_ref=None,
):
    """MILP. Variables per hour t: P_t (kW), R_t (renewable used, kW), z_t (on/off).
    minimise w*Cost/cost_ref + (1-w)*CO2/co2_ref
    s.t. hydrogen >= demand, min/max load, ramp limit, R_t <= generation.
    """
    gen_arr = np.asarray(gen_kwh, dtype=float)
    T = 24
    n = 3 * T  # [P(0..T), R(T..2T), z(2T..3T)]

    cost_c = np.concatenate([price, ren_price - price, np.zeros(T)])
    co2_c = np.concatenate([ef, -ef, np.zeros(T)])

    cost_ref = 1.0 if not cost_ref else cost_ref
    co2_ref = 1.0 if not co2_ref else co2_ref

    c = w * cost_c / cost_ref + (1 - w) * co2_c / co2_ref

    A, lo, hi = [], [], []

    def add(row, l, h):
        A.append(row)
        lo.append(l)
        hi.append(h)

    # Hydrogen demand constraint: sum(P_t) / kwh_per_kg >= demand_kg
    row = np.zeros(n)
    row[:T] = 1.0 / kwh_per_kg
    add(row, demand_kg, np.inf)

    for t in range(T):
        # P_t - cap_kw * z_t <= 0
        r = np.zeros(n)
        r[t] = 1.0
        r[2 * T + t] = -cap_kw
        add(r, -np.inf, 0.0)

        # P_t - min_load * cap_kw * z_t >= 0
        r = np.zeros(n)
        r[t] = 1.0
        r[2 * T + t] = -min_load * cap_kw
        add(r, 0.0, np.inf)

        # P_t - R_t >= 0  =>  R_t <= P_t
        r = np.zeros(n)
        r[t] = 1.0
        r[T + t] = -1.0
        add(r, 0.0, np.inf)

        # Ramp limit: -ramp*cap_kw <= P_t - P_{t-1} <= ramp*cap_kw
        if t > 0:
            r = np.zeros(n)
            r[t] = 1.0
            r[t - 1] = -1.0
            add(r, -ramp * cap_kw, ramp * cap_kw)

    ub = np.concatenate([np.full(T, cap_kw), gen_arr, np.ones(T)])
    integrality = np.concatenate([np.zeros(2 * T), np.ones(T)])

    res = milp(
        c,
        constraints=LinearConstraint(np.array(A), lo, hi),
        integrality=integrality,
        bounds=Bounds(np.zeros(n), ub),
    )

    if not res.success:
        return None

    P, R = res.x[:T], res.x[T : 2 * T]
    return summarize(P, R, price, ef, ren_price, kwh_per_kg)


def summarize(P, R, price, ef, ren_price, kwh_per_kg):
    G = np.maximum(P - R, 0.0)
    cost = float((G * price).sum() + (R * ren_price).sum())
    co2 = float((G * ef).sum())
    kg = float(P.sum() / kwh_per_kg)
    return dict(
        P=P,
        R=R,
        G=G,
        cost=cost,
        co2=co2,
        kg=kg,
        cost_per_kg=cost / max(kg, 1e-9),
        co2_per_kg=co2 / max(kg, 1e-9),
    )


def baseline_24x7(
    gen_kwh, demand_kg, kwh_per_kg=52.0, price=GRID_PRICE, ef=GRID_EF, ren_price=RENEWABLE_PRICE
):
    gen_arr = np.asarray(gen_kwh, dtype=float)
    P = np.full(24, demand_kg * kwh_per_kg / 24.0)
    R = np.minimum(P, gen_arr)
    return summarize(P, R, price, ef, ren_price, kwh_per_kg)


def pareto(gen_kwh, demand_kg, **kw):
    """Sweep w from 0 (pure CO2) to 1 (pure cost). Returns list of (w, result)."""
    base = baseline_24x7(gen_kwh, demand_kg, kwh_per_kg=kw.get("kwh_per_kg", 52.0))
    out = []
    for w in np.linspace(0, 1, 11):
        r = solve(gen_kwh, demand_kg, w=w, cost_ref=base["cost"], co2_ref=base["co2"], **kw)
        if r:
            out.append((float(w), r))
    return out, base