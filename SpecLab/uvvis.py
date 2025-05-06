import pandas as pd
from pathlib import Path

def load_uvvis_folder(folder_path: str) -> pd.DataFrame:
    """
    Loads and merges UV-Vis .txt files from a folder into a single DataFrame.

    Each file must have a header row to skip and two columns: wavelength and intensity.
    The function assumes all files share the same wavelength format and merges them by this column.

    Args:
        folder_path (str): Path to the folder containing the .txt files.

    Returns:
        pd.DataFrame: A DataFrame with one 'wavelength' column and one column per file.
    """
    files = sorted(Path(folder_path).glob("*.txt"))
    merged_df = None

    for file in files:
        df = pd.read_csv(file, skiprows=1)
        df.columns = ["wavelength", file.stem]
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="wavelength", how="outer")

    return merged_df.sort_values("wavelength").reset_index(drop=True)

