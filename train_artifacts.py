from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from service import INITIAL_COLS_TO_DROP, TARGET_COL


ARTIFACTS_DIR = Path("artifacts")
DATA_PATH = Path("fpl_final.csv")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    train_df = df.copy()
    train_df.drop(columns=INITIAL_COLS_TO_DROP, inplace=True, errors="ignore")

    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        random_state=44,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train_scaled, y_train)

    model.save_model(ARTIFACTS_DIR / "xgb_model.json")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    metadata = {"feature_columns": X.columns.tolist()}
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved artifacts to {ARTIFACTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
