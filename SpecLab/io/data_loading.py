import pandas as pd
from pathlib import Path
from typing import Union, Optional, List

def load_xy_folder(folder_path: Union[str, Path], 
                   x_label: str = "wavelength", 
                   skiprows: int = 0,
                   extensions: tuple = (".txt", ".csv"),
                   filter: Optional[List[str]] = None,
                   verbose: bool = False) -> pd.DataFrame:
    """
    Loads and merges multiple two-column text or CSV files from a folder (recursively) into a single DataFrame.

    Args:
        folder_path (Union[str, Path]): Path to the main folder containing files or subfolders.
        x_label (str): Name to assign to the x-axis column (default is "wavelength").
        skiprows (int): Number of rows to skip at the beginning of each file.
        extensions (tuple): File extensions to consider (default: (".txt", ".csv")).
        filter (Optional[List[str]]): List of strings. Only files whose name contains at least one of these strings
            will be loaded. If None or empty, all files are loaded.
        verbose (bool): If True, prints detailed information about file processing.

    Returns:
        pd.DataFrame: DataFrame with merged x-values and one y-column per file (named after filename).
    """
    folder_path = Path(folder_path)
    files = sorted([f for f in folder_path.rglob("*") if f.suffix in extensions])
    total_files = len(files)
    if filter:
        filter_lower = [word.lower() for word in filter]
        files = [f for f in files if any(word in f.name.lower() for word in filter_lower)]
    filtered_files = len(files)

    merged_df = None
    processed = 0
    skipped_malformed = 0
    skipped_error = 0

    for file in files:
        try:
            df = pd.read_csv(file, skiprows=skiprows, na_values=["", " ", "NA", "nan"])
            if df.shape[1] != 2:
                skipped_malformed += 1
                if verbose:
                    print(f"Skipping {file.name}: does not have exactly 2 columns.")
                continue  # skip malformed files
            df.columns = [x_label, file.stem]
            df[file.stem] = pd.to_numeric(df[file.stem], errors="coerce")
            df[x_label] = pd.to_numeric(df[x_label], errors="coerce")
            df = df.dropna(subset=[x_label, file.stem])
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=x_label, how="outer")
            processed += 1
        except Exception as e:
            skipped_error += 1
            if verbose:
                print(f"Error loading {file.name}: {e}")
            continue

    if verbose:
        print(f"Total files found: {total_files}")
        if filter:
            print(f"Files after filter: {filtered_files}")
        print(f"Files successfully processed: {processed}")
        print(f"Files skipped (malformed): {skipped_malformed}")
        print(f"Files skipped (error): {skipped_error}")

    if merged_df is not None:
        return merged_df.sort_values(x_label).reset_index(drop=True)
    else:
        raise ValueError("No valid files found.")
