import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def substract_baseline(
    df: pd.DataFrame,
    baseline_col: str, 
    target_cols: list[str] = None
) -> pd.DataFrame:
    """"
    Subtracts a baseline column from selected columns in the DataFrame.

    If no target columns are provided, the function will apply the correction
    to all columns except the x-axis column and the baseline itself.

    Args:
        df (pd.DataFrame): DataFrame containing spectral data.
        baseline_col (str): Name of the column to use as baseline.
        target_cols (list[str], optional): Names of columns to subtract the baseline from.
            If None, all columns except the first and the baseline are used.

    Returns:
        pd.DataFrame: A copy of the original DataFrame with corrected target columns.
    """
    corrected_df = df.copy()

    if target_cols is None:
        x_col = df.columns[0]
        target_cols = [col for col in df.columns if col not in (x_col, baseline_col)]

    for col in target_cols:
        corrected_df[col] = corrected_df[col] - corrected_df[baseline_col]
    
    return corrected_df

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
