import pandas as pd
import numpy as np

def enrich_date_features(df: pd.DataFrame):
    df = df.copy()
    date_cols = []

    for col in df.columns:
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                date_cols.append(col)
            except Exception:
                pass

    for col in date_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_dow"] = df[col].dt.dayofweek
        df[f"{col}_ord"] = df[col].map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)

    return df, date_cols
