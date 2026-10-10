"""Phase 5 - Natural Language Processing (NLP) for Market Sentiment.

Historical price data lacks context (earnings reports, geopolitics, product launches).
This module fetches real-time news headlines using yfinance and scores their
sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).

Since we can only easily fetch *recent* news (not 10 years of history), we use
this to provide a real-time sentiment gauge to accompany our price forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

@dataclass
class NewsItem:
    headline: str
    publisher: str
    timestamp: datetime
    sentiment_score: float  # -1.0 (very negative) to 1.0 (very positive)
    sentiment_label: str

def get_recent_news_sentiment(ticker: str) -> list[NewsItem]:
    """Fetch recent news and Reddit hype for a ticker and analyze sentiment."""
    import urllib.request
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return []

    analyzer = SentimentIntensityAnalyzer()
    results = []
    
    # 1. Google News RSS Fetch
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        xml_data = urllib.request.urlopen(req, timeout=10).read()
        root = ET.fromstring(xml_data)
        news_data = root.findall('.//item')[:15]
        
        for item in news_data:
            headline_elem = item.find("title")
            if headline_elem is None or not headline_elem.text:
                continue
            headline = headline_elem.text
            
            source_elem = item.find("source")
            publisher = source_elem.text if source_elem is not None else "News"
            
            pubDate_elem = item.find("pubDate")
            dt = parsedate_to_datetime(pubDate_elem.text) if (pubDate_elem is not None and pubDate_elem.text) else datetime.now()

            score = analyzer.polarity_scores(headline)["compound"]
            label = "POSITIVE" if score > 0.15 else "NEGATIVE" if score < -0.15 else "NEUTRAL"
            results.append(NewsItem(headline, publisher, dt, score, label))
    except Exception:
        pass

    # 2. Reddit Hype Fetch (Phase 2 Upgrade)
    try:
        import praw
        import streamlit as st
        
        client_id = st.secrets.get("REDDIT_CLIENT_ID")
        client_secret = st.secrets.get("REDDIT_CLIENT_SECRET")
        
        if client_id and client_secret:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent="python:stock-predictor:1.0"
            )
            
            subs = "IndianStreetBets+IndiaInvestments" if ticker.endswith((".NS", ".BO")) else "wallstreetbets+stocks+investing"
            search_query = ticker.split('.')[0] # Remove .NS for search
            
            # Search for recent posts about this ticker
            for submission in reddit.subreddit(subs).search(search_query, sort='new', time_filter='week', limit=10):
                score = analyzer.polarity_scores(submission.title)["compound"]
                label = "POSITIVE" if score > 0.15 else "NEGATIVE" if score < -0.15 else "NEUTRAL"
                dt = datetime.fromtimestamp(submission.created_utc)
                
                results.append(NewsItem(
                    headline=submission.title,
                    publisher=f"Reddit (r/{submission.subreddit.display_name})",
                    timestamp=dt,
                    sentiment_score=score,
                    sentiment_label=label,
                ))
    except Exception as e:
        pass # Silently fail if reddit API has issues or keys are missing

    # Sort combined results by newest first
    results.sort(key=lambda x: x.timestamp, reverse=True)
    return results

def aggregate_sentiment(news: list[NewsItem]) -> tuple[float, str]:
    """Calculate the average market sentiment from a list of news items."""
    if not news:
        return 0.0, "UNKNOWN"
    
    avg_score = sum(item.sentiment_score for item in news) / len(news)
    if avg_score > 0.15:
        return avg_score, "BULLISH"
    elif avg_score < -0.15:
        return avg_score, "BEARISH"
    return avg_score, "NEUTRAL"
