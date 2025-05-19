import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def calibration_curve(
    df: pd.DataFrame,
    x_data: list[float],
    target_x: float = 0,
    target_cols: list[str] = None
) -> tuple[plt.figure, float, float, float]:

    if target_cols is None:
        x_col = df.columns[0]
        target_cols = [col for col in df.columns if col != x_col]

    if target_x == 0:
        y_data = [df[col].max() for col in target_cols]
    else:
        idx = (df[x_col] - target_x).abs().idxmin()
        y_data = [df[col].iloc[idx] for col in target_cols]
    X = np.array(x_data).reshape(-1, 1)
    y = np.array(y_data)

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Data", color="blue")
    ax.plot(X, predictions, label=f"Lineal regression\n$R^2$ = {r2:.3f}", color="red")
    ax.legend()
    ax.grid(True)

    return fig, {"slope": slope, "intercept": intercept, "r2": r2}       

