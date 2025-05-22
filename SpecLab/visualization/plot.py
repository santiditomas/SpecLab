import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def plot(
    df: pd.DataFrame,
    method: str = "classic",
    x_col: str = "wavelength",
    target_cols: list[str] = None,
    xlim: list[float] = [0, 0],
    ylim: list[float] = [0, 0],
    x_label: str = "wavelength",
    y_label: str = "Value"
):
    if target_cols is None:
        target_cols = [col for col in df.columns if col != x_col]

    if method == "classic":
        fig, ax = plt.subplots()
        for col in target_cols:
            ax.plot(df[x_col], df[col], label=col)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_label)
        ax.legend()

        if xlim != [0, 0]:
            ax.set_xlim(xlim)
        if ylim != [0, 0]:
            ax.set_ylim(ylim)

        plt.tight_layout()
        plt.show()
        return fig

    elif method == "interactive":
        df_long = df.melt(id_vars=[x_col], value_vars=target_cols, 
                          var_name='variable', value_name='value')
        fig = px.line(df_long, x=x_col, y='value', color='variable')
        if xlim != [0, 0]:
            fig.update_xaxes(range=xlim)
        if ylim != [0, 0]:
            fig.update_yaxes(range=ylim)
        fig.show()
        return fig

    else:
        raise ValueError(f"Unknown method: {method}")
    


