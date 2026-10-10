import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for streamlit

from stock_predictor import data, features, models, evaluation, forecast, visualize, nlp, trading, report, scanner, alerts
from stock_predictor.config import Config, OUTPUT_DIR

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

st.title("📈 Advanced Stock Price Predictor")
st.markdown("Machine Learning Pipeline (Linear Regression, Random Forest, LSTM) & NLP Sentiment Analysis")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    TICKERS = {
        "AAPL": "AAPL (Apple)",
        "TSLA": "TSLA (Tesla)",
        "MSFT": "MSFT (Microsoft)",
        "GOOG": "GOOG (Google)",
        "AMZN": "AMZN (Amazon)",
        "NVDA": "NVDA (Nvidia)",
        "RELIANCE.NS": "RELIANCE.NS (Reliance Industries)",
        "TCS.NS": "TCS.NS (Tata Consultancy Services)",
        "TATAMOTORS.NS": "TATAMOTORS.NS (Tata Motors)",
        "HDFCBANK.NS": "HDFCBANK.NS (HDFC Bank)",
        "ZOMATO.NS": "ZOMATO.NS (Zomato)",
        "INFY.NS": "INFY.NS (Infosys)",
        "OTHER": "➕ Other"
    }
    
    selected_option = st.selectbox("Ticker Symbol", options=list(TICKERS.keys()), format_func=lambda x: TICKERS[x])
    
    if selected_option == "OTHER":
        ticker = st.text_input("Enter Other Ticker (e.g. META, BTC-USD)", value="").upper()
    else:
        ticker = selected_option
        
    forecast_days = st.slider("Forecast Days", min_value=5, max_value=60, value=30)
    st.markdown("---")
    run_btn = st.button("Predict", type="primary")
    
    st.markdown("---")
    st.subheader("🏆 Today's Market Leaderboard")
    if st.button("Scan Top Stocks"):
        with st.spinner("Scanning Market..."):
            leaderboard = scanner.get_market_leaderboard(list(TICKERS.keys())[:-1]) # exclude 'OTHER'
            if not leaderboard.empty:
                st.dataframe(
                    leaderboard.style.format({
                        "Last Price": "{:.2f}", 
                        "% Change (Today)": "{:+.2f}%", 
                        "Volume": "{:,.0f}"
                    }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['% Change (Today)']),
                    use_container_width=True
                )
                
    st.markdown("---")
    st.subheader("🔔 Telegram Alerts")
    if st.button("Send Test Alert"):
        success = alerts.send_telegram_alert(f"🤖 *AI Stock Predictor*\nTest alert successful! Ready to send daily trading signals.")
        if success:
            st.success("Alert sent to Telegram!")
        else:
            st.error("Missing Bot Token! Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to secrets.toml")

if run_btn:
    if not ticker.strip():
        st.error("⚠️ Please enter a valid Ticker Symbol before running the pipeline.")
    else:
        cfg = Config(ticker=ticker, forecast_days=forecast_days, show_plot=False)
    
        with st.spinner(f"Fetching Data for {ticker}..."):
            raw = data.fetch_data(ticker, cfg.start, cfg.end, cfg.refresh)
            prices, clean_report = data.clean_data(raw)
            
        with st.spinner("Engineering Technical Features..."):
            feats = features.add_features(prices)
            dataset = features.make_supervised(feats)
            test_start = prices.index[-1] - pd.Timedelta(days=365)
            train, test = features.split_by_date(dataset, test_start)
            
        with st.spinner("Training Machine Learning Models (LR, RF, LSTM)..."):
            trained, predicted_returns = {}, {}
            for name, model in models.build_models().items():
                trained[name] = models.fit(model, train)
                predicted_returns[name] = models.predict_returns(model, test)
                
        with st.spinner("Evaluating & Forecasting..."):
            names = list(trained)
            backtest = evaluation.build_backtest_frame(test, predicted_returns)
            metrics = evaluation.metrics_table(backtest, names)
            equity = evaluation.strategy_equity(backtest, names)
            best_ml = metrics.loc[names, "RMSE"].idxmin()
            
            final_model = models.fit(models.build_models()[best_ml], dataset)
            error_std = float((backtest["Actual_Return"] - backtest[f"PredReturn_{best_ml}"]).std())
            future = forecast.forecast_future(final_model, prices, forecast_days, error_std)
            
            vix = data.get_vix()
            funds = data.get_fundamentals(ticker)
            
            # Save results to session state so it doesn't reload on every click
            st.session_state["pipeline_results"] = {
                "ticker": ticker,
                "prices": prices,
                "feats": feats,
                "backtest": backtest,
                "metrics": metrics,
                "equity": equity,
                "future": future,
                "best_ml": best_ml,
                "test_start": test_start,
                "raw_data": raw,
                "vix": vix,
                "funds": funds
            }

if "pipeline_results" in st.session_state:
    res = st.session_state["pipeline_results"]
    ticker = res["ticker"]
    prices = res["prices"]
    feats = res["feats"]
    backtest = res["backtest"]
    metrics = res["metrics"]
    equity = res["equity"]
    future = res["future"]
    best_ml = res["best_ml"]
    test_start = res["test_start"]
    raw_data = res["raw_data"]
    vix = res.get("vix", 0.0)
    funds = res.get("funds", {})

    # Generate interactive Plotly dashboard
    from stock_predictor import visualize_plotly
    fig = visualize_plotly.plot_interactive_dashboard(
        ticker, feats, backtest, metrics, equity, future, best_ml, test_start
    )

    # Layout Results
    st.success(f"Pipeline complete! Best performing model on test data: **{best_ml}**")
    
    # PDF Report Download
    pdf_bytes = report.generate_pdf_report(
        ticker=ticker,
        current_price=prices["Close"].iloc[-1],
        future_price=future["Predicted_Close"].iloc[-1],
        best_ml=best_ml,
        hype_trend="UNKNOWN", # Will be updated dynamically if needed, or we just pass a simple string
        sentiment_score=0.0,
        vix=vix,
        funds=funds
    )
    
    st.download_button(
        label="📄 Download AI Report (PDF)",
        data=pdf_bytes,
        file_name=f"{ticker}_AI_Report.pdf",
        mime="application/pdf",
        type="primary"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Price Forecast & Backtest Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("🏢 Company Fundamentals & VIX")
        if funds:
            c1, c2 = st.columns(2)
            c1.metric("P/E Ratio", funds.get("PE", "N/A"))
            c2.metric("EPS", funds.get("EPS", "N/A"))
            st.caption(f"Industry: {funds.get('Industry', 'N/A')}")
            
        vix_color = "red" if vix > 20 else "green" # High VIX = Fear
        vix_status = "Fear (High Volatility)" if vix > 20 else "Greed (Stable Market)"
        st.markdown(f"**Market VIX Index (Fear/Greed):** :{vix_color}[{vix:.2f}] - {vix_status}")
        
        st.write("---")
        st.subheader("📰 Real-time NLP Sentiment & Hype")
        
        hype = nlp.get_wikipedia_hype(ticker)
        if hype["views"] > 0:
            st.markdown(f"**Wikipedia Search Hype (Last 7 Days)**: {hype['views']:,} views")
            if "RISING" in hype["trend"]:
                st.success(f"Trend: {hype['trend']} (+{hype['change']:.1f}%)")
            elif "FALLING" in hype["trend"]:
                st.error(f"Trend: {hype['trend']} ({hype['change']:.1f}%)")
            else:
                st.info(f"Trend: {hype['trend']} ({hype['change']:.1f}%)")
                
        st.write("---")
        
        news_items = nlp.get_recent_news_sentiment(ticker)
        avg_score = 0.0
        if news_items:
            avg_score, sentiment = nlp.aggregate_sentiment(news_items)
            st.metric("Market Mood", sentiment, f"{avg_score:+.2f} VADER Score")
            for item in news_items[:7]:
                # Streamlit markdown colors
                color = "green" if item.sentiment_label == "POSITIVE" else "red" if item.sentiment_label == "NEGATIVE" else "gray"
                st.markdown(f"- **:{color}[{item.sentiment_label} ({item.sentiment_score:+.2f})]** [{item.timestamp.strftime('%m-%d %H:%M')}] {item.headline}")
        else:
            st.info("No recent news found.")
            
        st.subheader("📉 Test Year Performance")
        st.dataframe(metrics.style.format("{:.3f}"), use_container_width=True)
        
    st.write("---")
    
    # PHASE 3: Virtual Trading Bot
    st.header("🤖 AI Virtual Trading Bot (Paper Trading)")
    st.markdown("Yeh bot virtual money use karke AI ki prediction ke hisaab se trade karta hai. Aapko starting mein **1,00,000 (Virtual Money)** mila hai!")
    
    current_price = prices["Close"].iloc[-1]
    recommendation = trading.get_ai_recommendation(future, current_price, hype.get("trend", "STABLE"), avg_score)
    
    col_bot1, col_bot2 = st.columns([1, 2])
    
    with col_bot1:
        st.subheader("💡 AI Suggestion")
        if "BUY" in recommendation:
            st.success(f"**{recommendation}**")
        elif "SELL" in recommendation:
            st.error(f"**{recommendation}**")
        else:
            st.warning(f"**{recommendation}**")
            
        st.metric(label=f"Current Price of {ticker}", value=f"{current_price:,.2f}")
        
        trade_qty = st.number_input("Shares to Trade", min_value=1, value=10)
        
        c1, c2 = st.columns(2)
        if c1.button("🟢 BUY", use_container_width=True):
            success, msg = trading.execute_trade(ticker, "BUY", trade_qty, current_price)
            if success:
                st.success(msg)
            else:
                st.error(msg)
                
        if c2.button("🔴 SELL", use_container_width=True):
            success, msg = trading.execute_trade(ticker, "SELL", trade_qty, current_price)
            if success:
                st.success(msg)
            else:
                st.error(msg)
                
    with col_bot2:
        st.subheader("💼 Your Portfolio")
        pf = trading.get_portfolio()
        st.metric("Available Balance", f"{pf['balance']:,.2f}")
        
        if pf["positions"]:
            import pandas as pd
            pos_df = pd.DataFrame.from_dict(pf["positions"], orient="index")
            pos_df["Current Price"] = current_price # (Simplified: only shows current price for active ticker)
            pos_df["Profit/Loss"] = (pos_df["Current Price"] - pos_df["avg_price"]) * pos_df["shares"]
            st.dataframe(pos_df.style.format("{:.2f}").map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['Profit/Loss']), use_container_width=True)
        else:
            st.info("No active positions. Buy some shares!")
            
        if pf["history"]:
            with st.expander("Show Trade History"):
                st.dataframe(pf["history"][::-1]) # Show newest first
