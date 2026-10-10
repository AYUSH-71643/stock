import json
import os
from pathlib import Path
from datetime import datetime

PORTFOLIO_FILE = Path("data/portfolio.json")

def _load_portfolio() -> dict:
    if not PORTFOLIO_FILE.exists():
        PORTFOLIO_FILE.parent.mkdir(exist_ok=True)
        default_portfolio = {
            "balance": 100000.0,  # $100k or ₹100k starting virtual money
            "positions": {},      # {"AAPL": {"shares": 10, "avg_price": 150.0}}
            "history": []         # List of trades
        }
        _save_portfolio(default_portfolio)
        return default_portfolio
    
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"balance": 100000.0, "positions": {}, "history": []}

def _save_portfolio(data: dict):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)

def get_portfolio() -> dict:
    return _load_portfolio()

def execute_trade(ticker: str, action: str, shares: float, price: float) -> tuple[bool, str]:
    """Execute a virtual paper trade."""
    pf = _load_portfolio()
    cost = shares * price
    
    if action == "BUY":
        if pf["balance"] < cost:
            return False, f"Insufficient balance! Need {cost:.2f}, but have {pf['balance']:.2f}"
            
        pf["balance"] -= cost
        
        # Update positions
        if ticker not in pf["positions"]:
            pf["positions"][ticker] = {"shares": shares, "avg_price": price}
        else:
            old_shares = pf["positions"][ticker]["shares"]
            old_price = pf["positions"][ticker]["avg_price"]
            new_shares = old_shares + shares
            # Weighted average price
            new_avg_price = ((old_shares * old_price) + (shares * price)) / new_shares
            pf["positions"][ticker]["shares"] = new_shares
            pf["positions"][ticker]["avg_price"] = new_avg_price
            
    elif action == "SELL":
        if ticker not in pf["positions"] or pf["positions"][ticker]["shares"] < shares:
            return False, "Not enough shares to sell!"
            
        pf["balance"] += cost
        pf["positions"][ticker]["shares"] -= shares
        
        if pf["positions"][ticker]["shares"] <= 0:
            del pf["positions"][ticker]
            
    else:
        return False, "Invalid action."
        
    # Log history
    pf["history"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "price": price,
        "total_value": cost
    })
    
    _save_portfolio(pf)
    return True, f"Successfully {action} {shares} shares of {ticker} at {price:.2f}"

def get_ai_recommendation(forecast_df, current_price, hype_trend="STABLE", news_score=0.0) -> str:
    """Analyze forecast and sentiment to recommend BUY, SELL, or HOLD."""
    if forecast_df.empty:
        return "HOLD"
        
    future_price = forecast_df["Predicted_Close"].iloc[-1]
    percent_change = ((future_price - current_price) / current_price) * 100
    
    # Base recommendation from ML models (Technicals & Price)
    if percent_change > 3.0:
        base_rec = "BUY"
    elif percent_change < -3.0:
        base_rec = "SELL"
    else:
        base_rec = "HOLD"
        
    # Phase 2 Blend: Adjust based on Public Hype & News Sentiment
    # If ML says BUY, but Hype is Falling AND Sentiment is Negative -> Downgrade to HOLD
    if base_rec == "BUY":
        if "FALLING" in hype_trend and news_score < 0:
            return "HOLD (ML says UP, but Hype & News are crashing!)"
        return f"BUY (ML expects {percent_change:+.1f}%)"
        
    # If ML says SELL, but Hype is Rising AND Sentiment is Positive -> Upgrade to HOLD
    elif base_rec == "SELL":
        if "RISING" in hype_trend and news_score > 0:
            return "HOLD (ML says DOWN, but Hype & News are very strong!)"
        return f"SELL (ML expects {percent_change:+.1f}%)"
        
    else:
        # If ML is unsure, let Sentiment decide slightly
        if "RISING" in hype_trend and news_score > 0.5:
            return "BUY (ML is neutral, but strong Hype & News!)"
        elif "FALLING" in hype_trend and news_score < -0.5:
            return "SELL (ML is neutral, but Hype & News are crashing!)"
            
        return f"HOLD (Market is flat, expects {percent_change:+.1f}%)"
