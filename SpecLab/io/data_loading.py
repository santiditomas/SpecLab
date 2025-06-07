import pandas as pd
from pathlib import Path
from typing import Union

def load_xy_folder(folder_path: Union[str, Path], 
                   x_label: str = "wavelength", 
                   skiprows: int = 0,
                   extensions: tuple = (".txt", ".csv")) -> pd.DataFrame:
    """
    Loads and merges multiple two-column text or CSV files from a folder (recursively) into a single DataFrame.

    Args:
        folder_path (Union[str, Path]): Path to the main folder containing files or subfolders.
        x_label (str): Name to assign to the x-axis column (default is "wavelength").
        skiprows (int): Number of rows to skip at the beginning of each file.
        extensions (tuple): File extensions to consider (default: (".txt", ".csv")).

    Returns:
        pd.DataFrame: DataFrame with merged x-values and one y-column per file (named after filename).
    """
    folder_path = Path(folder_path)
    files = sorted([f for f in folder_path.rglob("*") if f.suffix in extensions])

    merged_df = None

    for file in files:
        try:
            df = pd.read_csv(file, skiprows=skiprows, na_values=["", " ", "NA", "nan"])
            if df.shape[1] != 2:
                continue  # skip malformed files
            df.columns = [x_label, file.stem]
            df[file.stem] = pd.to_numeric(df[file.stem], errors="coerce")
            df[x_label] = pd.to_numeric(df[x_label], errors="coerce")
            df = df.dropna(subset=[x_label, file.stem])
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=x_label, how="outer")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    if merged_df is not None:
        return merged_df.sort_values(x_label).reset_index(drop=True)
    else:
        raise ValueError("No valid files found.")
