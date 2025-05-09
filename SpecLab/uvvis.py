import pandas as pd

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