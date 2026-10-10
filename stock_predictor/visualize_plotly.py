import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from .evaluation import BASELINE_NAME

MODEL_COLORS = {
    "Linear Regression": "#58a6ff",
    "Random Forest": "#3fb950",
    "LSTM (Neural Net)": "#ff7b72",
    BASELINE_NAME: "#8b949e",
    "Buy & Hold": "#e6edf3",
}

def plot_interactive_dashboard(
    ticker: str,
    features: pd.DataFrame,
    backtest: pd.DataFrame,
    metrics: pd.DataFrame,
    equity: pd.DataFrame,
    forecast: pd.DataFrame,
    forecast_model: str,
    test_start: pd.Timestamp,
) -> go.Figure:
    
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"colspan": 2}, None],
               [{}, {}],
               [{}, {}]],
        subplot_titles=(
            "Price History, Moving Averages & Forecast",
            "Backtest: Predictions vs Actual (Test Set)",
            f"Forecast: Next {len(forecast)} Days ({forecast_model})",
            "RSI (14) - Momentum",
            "Strategy Backtest: $1 Invested (Long when model predicts up)"
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )

    # 1. Main Price Chart
    fig.add_trace(go.Scatter(x=features.index, y=features["Close"], name="Close", line=dict(color="#e6edf3", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=features.index, y=features["SMA_50"], name="SMA 50", line=dict(color="#f2cc60", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=features.index, y=features["SMA_200"], name="SMA 200", line=dict(color="#ff7b72", width=1)), row=1, col=1)
    
    # Add Forecast to main chart
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["Predicted_Close"], name="Forecast", line=dict(color="#d2a8ff", width=2)), row=1, col=1)
    
    # Highlight Test Area
    fig.add_vrect(x0=test_start, x1=backtest.index[-1], fillcolor="#1f6feb", opacity=0.1, layer="below", line_width=0, row=1, col=1, annotation_text="Test Period")

    # 2. Backtest Chart
    fig.add_trace(go.Scatter(x=backtest.index, y=backtest["Actual_Close"], name="Actual", line=dict(color="#e6edf3", width=2)), row=2, col=1)
    for model_name in [m for m in metrics.index if m != BASELINE_NAME]:
        fig.add_trace(go.Scatter(x=backtest.index, y=backtest[f"Pred_{model_name}"], name=f"{model_name} (RMSE {metrics.loc[model_name, 'RMSE']:.2f})", line=dict(color=MODEL_COLORS.get(model_name), width=1)), row=2, col=1)

    # 3. Forecast Zoom
    recent = features["Close"].tail(60)
    fig.add_trace(go.Scatter(x=recent.index, y=recent, name="Recent Close", line=dict(color="#e6edf3", width=1.5)), row=2, col=2)
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["Predicted_Close"], name="Future Forecast", line=dict(color="#d2a8ff", width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["Upper_95"], name="Upper Bound", line=dict(color="#d2a8ff", width=0), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["Lower_95"], name="Lower Bound", line=dict(color="#d2a8ff", width=0), fill='tonexty', fillcolor='rgba(210, 168, 255, 0.2)', showlegend=False), row=2, col=2)

    # 4. RSI
    rsi = features["RSI_14"].loc[features.index >= test_start - pd.Timedelta(days=180)]
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#f2cc60", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff7b72", row=3, col=1, annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="#3fb950", row=3, col=1, annotation_text="Oversold")
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # 5. Equity Curve
    for name in equity.columns:
        fig.add_trace(go.Scatter(x=equity.index, y=equity[name], name=f"{name} ({(equity[name].iloc[-1] - 1) * 100:+.1f}%)", line=dict(color=MODEL_COLORS.get(name, "#ffffff"), width=1.5)), row=3, col=2)
    fig.add_hline(y=1, line_color="#30363d", row=3, col=2)

    fig.update_layout(
        height=1000,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        hovermode="x unified",
        title_text=f"<b>{ticker.upper()}</b> Stock Analysis (Interactive)",
        title_x=0.5,
        title_font_size=20
    )
    
    return fig
