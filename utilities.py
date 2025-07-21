from typing import Any, Optional, TypeVar
import polars as pl
import plotly.express as px
from plotly.graph_objects import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

IntoFrameT = TypeVar("IntoFrameT", pl.DataFrame, pl.LazyFrame)

def _decompose_single_series(values: pl.Series, model: str, period: int) -> pl.Series:
    decomposition = seasonal_decompose(values, model=model, period=period, extrapolate_trend=1)
    
    result = pl.DataFrame({
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.resid
    }).select(pl.struct(pl.col("*")).alias("components")).get_column("components")
    return result


def decompose(df: IntoFrameT, model: str = "multiplicative", period: int = 12, id_col: str = "unique_id", time_col: str = "ds", target_col: str = "y") -> IntoFrameT:
    """
    Decomposes the time series into `trend`, `seasonal`, and `residual` components and adds them to the DataFrame.
    CAUTION: Do not use this function to create forecasting features, because the decomposition will leak future information into the past!

    Parameters
    ----------
    df : IntoFrameT
        DataFrame containing the time series data.
    model : str, optional
        Type of decomposition model to use, either 'additive' or 'multiplicative'.
    period : int, optional
        The number of periods in a complete cycle (e.g., 12 for monthly data).
    id_col : str, optional
        Column name for the unique identifier of each time series.
    time_col : str, optional
        Column name for the time index.
    target_col : str, optional
        Column name for the target variable to be decomposed.

    Returns
    -------
    IntoFrameT
        DataFrame with decomposed components.
    """
    return (
        df
        .sort(time_col)
        .with_columns(
            pl.col(target_col).map_batches(
                function=lambda x: _decompose_single_series(x, model=model, period=period),
            ).over(id_col).alias("components")
        )
        .unnest("components")
    )


def plot_components(df: pl.DataFrame, unique_id: Optional[Any], id_col: str = "unique_id", time_col: str = "ds", target_col: str = "y") -> Figure:
    """
    Plots the components of the time series. Expects the columns `trend`, `seasonal`, and `residual` to be present in the DataFrame.
    See `decompose` function for details on how to create these columns.

    Parameters
    ----------
    df : IntoFrameT
        DataFrame containing the decomposed components.
    unique_id : Any, optional
        Unique identifier for the time series to be plotted. If None, a random time series will be chosen
    id_col : str, optional
        Column name for the unique identifier of each time series.
    time_col : str, optional
        Column name for the time index.
    target_col : str, optional
        Column name for the target variable to be plotted.
    """

    if not unique_id and df.height >= 1:
        unique_id = df.sample(1).get_column(id_col).item(0)

    plot_df = (
        df
        .filter(pl.col(id_col) == unique_id)
        .select(time_col, target_col, "trend", "seasonal", "residual")
        .sort(time_col)
        .unpivot(
            index=time_col,
            variable_name="component",
            value_name=target_col,
        )
    )

    fig = px.line(plot_df, x=time_col, y=target_col, facet_col="component", facet_col_wrap=1)
    fig.update_yaxes(matches=None)  # Ensure each subplot has its own y-axis scale
    return(fig)


def plot_seasonality(df: pl.DataFrame, unique_id: Any, id_col: str = "unique_id", period_expr: pl.Expr = pl.col("ds").dt.strftime("%b").alias("month")) -> Figure:
    """
    Plots the seasonality of the time series in a bar polar plot.
    """
    plot_df = (
        df
        .filter(pl.col(id_col).eq(unique_id))
        .group_by(period_expr, maintain_order=True)
        .agg(pl.col("seasonal").mean())
    )
    fig = px.bar_polar(plot_df, r="seasonal", theta=period_expr.meta.output_name(), color="seasonal", color_continuous_scale="bluered")
    fig.update_layout(polar = dict(radialaxis = dict(visible = False))) # hide radial axis
    return(fig)

def plot_seasonalities(df: pl.DataFrame, unique_ids: Optional[list] = None, id_col: str = "unique_id", period_expr: pl.Expr = pl.col("ds").dt.strftime("%b").alias("month"), facet_col_wrap: int = 4) -> Figure:
    """
    Plots the seasonality for multiple time series in a faceted bar polar plot.
    """
    if not unique_ids:
        unique_ids = df.get_column(id_col).unique().to_list()

    cols = min(len(unique_ids), facet_col_wrap)
    rows = (len(unique_ids) + cols - 1) // cols

    fig = make_subplots(
        cols=cols,
        rows=rows,
        specs=[[{"type": "barpolar"}] * cols]*rows,
        subplot_titles=unique_ids,
    )

    # add traces for each unique_id
    for idx, unique_id in enumerate(unique_ids):
        sub_fig = plot_seasonality(df, unique_id=unique_id, id_col=id_col, period_expr=period_expr)
        fig.add_trace(
            list(sub_fig.select_traces())[0],
            col=(idx % cols) + 1,
            row=(idx // cols) + 1,
        )

    # hide radial axis for all subplots
    for layout_key in fig.layout:
        if layout_key.startswith("polar"):
            fig.layout[layout_key].update(
                dict(radialaxis = dict(visible = False))
            )

    # move subplot titles to the left of each subplot
    for idx, annotation in enumerate(fig.layout.annotations):
        annotation["xanchor"] = "left"
        annotation["x"] = fig.layout[f"polar{idx+1}"]["domain"]["x"][0]

    # set color scale to blue-red and height depending on number of rows
    fig.update_layout(coloraxis_autocolorscale=False, coloraxis_colorscale=["blue", "red"], height=300 * rows)

    return(fig)