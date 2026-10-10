import yfinance as yf
import pandas as pd

def get_market_leaderboard(tickers: list[str]) -> pd.DataFrame:
    """Scan a list of tickers and return today's leaderboard."""
    data = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
                curr_close = hist['Close'].iloc[-1]
                pct_change = ((curr_close - prev_close) / prev_close) * 100
                data.append({
                    "Ticker": ticker,
                    "Last Price": curr_close,
                    "% Change (Today)": pct_change,
                    "Volume": hist['Volume'].iloc[-1]
                })
        except Exception:
            pass
            
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    df = df.sort_values(by="% Change (Today)", ascending=False).reset_index(drop=True)
    return df
