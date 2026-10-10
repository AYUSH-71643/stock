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

    # Sort combined results by newest first
    results.sort(key=lambda x: x.timestamp, reverse=True)
    return results

def get_wikipedia_hype(ticker: str) -> dict:
    """Fetch Wikipedia page views as a proxy for Retail Hype / Public Interest."""
    import urllib.request
    import json
    from datetime import datetime, timedelta
    
    WIKI_MAPPING = {
        "AAPL": "Apple_Inc.",
        "MSFT": "Microsoft",
        "GOOG": "Alphabet_Inc.",
        "AMZN": "Amazon",
        "NVDA": "Nvidia",
        "TSLA": "Tesla,_Inc.",
        "RELIANCE.NS": "Reliance_Industries",
        "TCS.NS": "Tata_Consultancy_Services",
        "TATAMOTORS.NS": "Tata_Motors",
        "HDFCBANK.NS": "HDFC_Bank",
        "ZOMATO.NS": "Zomato",
        "INFY.NS": "Infosys"
    }
    
    article = WIKI_MAPPING.get(ticker)
    if not article:
        return {"views": 0, "trend": "UNKNOWN", "change": 0.0}
        
    end = datetime.now()
    start = end - timedelta(days=14)
    
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{article}/daily/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
    req = urllib.request.Request(url, headers={'User-Agent': 'StockPredictorBot/1.0 (test@example.com)'})
    
    try:
        response = urllib.request.urlopen(req, timeout=5).read()
        data = json.loads(response)
        items = data.get("items", [])
        
        if len(items) < 14:
            return {"views": 0, "trend": "UNKNOWN", "change": 0.0}
            
        # Compare last 7 days vs previous 7 days
        recent_7 = sum(item["views"] for item in items[-7:])
        prev_7 = sum(item["views"] for item in items[-14:-7])
        
        if prev_7 == 0:
            return {"views": recent_7, "trend": "UNKNOWN", "change": 0.0}
            
        change = ((recent_7 - prev_7) / prev_7) * 100
        
        trend = "🔥 RISING HYPE" if change > 10 else ("📉 FALLING" if change < -10 else "⚖️ STABLE")
        
        return {
            "views": recent_7,
            "trend": trend,
            "change": change
        }
        
    except Exception:
        return {"views": 0, "trend": "UNKNOWN", "change": 0.0}

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
