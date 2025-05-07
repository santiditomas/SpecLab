import pandas as pd
from pathlib import Path
from typing import Union

def load_xy_folder(folder_path: Union[str, Path], x_label: str = "x", skiprows: int = 0) -> pd.DataFrame:
    """
    Loads and merges multiple two-column .txt files from a folder into a single DataFrame.

    Each file must contain exactly two columns: one representing the x-axis values and one representing
    the y-axis measurements. The x-axis column will be renamed according to `x_label`, and each y-axis 
    column will be named after the corresponding filename (without extension).

    Files are merged on the x-axis column using an outer join to preserve all values.

    Args:
        folder_path (Union[str, Path]): Path to the folder containing the .txt files.
        x_label (str): Name to assign to the x-axis column (default is "x").
        skiprows (int): Number of lines to skip at the beginning of each file (default is 0).

    Returns:
        pd.DataFrame: A DataFrame with the x-axis column and one column per file, merged by x values.
    """
    files = sorted(Path(folder_path).glob("*.txt"))
    merged_df = None

    for file in files:
        df = pd.read_csv(file, skiprows=skiprows)
        df.columns = ["wavelength", file.stem]
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="wavelength", how="outer")

    return merged_df.sort_values("wavelength").reset_index(drop=True)
