import requests
import streamlit as st

def send_telegram_alert(message: str) -> bool:
    """Send an alert to a Telegram Chat using bot token from secrets."""
    bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        return False
        
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        res = requests.post(url, json=payload)
        return res.status_code == 200
    except Exception:
        return False
