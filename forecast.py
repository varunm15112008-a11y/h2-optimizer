import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from data import make_training_set

FEATS = ["hour", "cloud", "wind"]


def train_models():
    df = make_training_set()
    tr_days, te_days = train_test_split(df.day.unique(), test_size=0.2, random_state=1)
    tr = df[df.day.isin(tr_days)]
    te = df[df.day.isin(te_days)]

    models, mae = {}, {}
    for tgt in ["solar_cf", "wind_cf"]:
        m = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=0)
        m.fit(tr[FEATS], tr[tgt])
        models[tgt] = m
        preds = np.clip(m.predict(te[FEATS]), 0, 1)
        mae[tgt] = mean_absolute_error(te[tgt], preds)

    return models, mae


def predict_day(models, cloud, wind):
    X = pd.DataFrame({
        "hour": np.arange(24),
        "cloud": np.broadcast_to(cloud, (24,)),
        "wind": np.broadcast_to(wind, (24,)),
    })
    s = np.clip(models["solar_cf"].predict(X), 0, 1)
    w = np.clip(models["wind_cf"].predict(X), 0, 1)
    return s, w