from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
import requests
import pickle
import numpy as np
import random

app = FastAPI()

DEXSCREENER_API = "https://api.dexscreener.com/latest/dex/search"

try:
    with open("solana_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except:
    model = None
    print("Model not found. Run train_model.py first.")

def fetch_token_data(symbol: str):
    try:
        resp = requests.get(DEXSCREENER_API, params={"q": symbol})
        data = resp.json()
        for pair in data.get("pairs", []):
            if pair.get("chainId") == "solana":
                return {
                    "symbol": pair["baseToken"]["symbol"],
                    "name": pair["baseToken"]["name"],
                    "price": float(pair["priceUsd"]),
                    "volume24h": float(pair["volume"]["h24"]),
                    "liquidityUsd": float(pair["liquidity"]["usd"])
                }
        return None
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def predict_trend(token_data):
    if not model or not token_data:
        return {"prediction": "unknown", "confidence": 0.0, "reasoning": "Model not loaded or data unavailable"}

    features = np.array([[token_data["price"], token_data["volume24h"], token_data["liquidityUsd"]]])
    pred_class = model.predict(features)[0]
    prob = model.predict_proba(features).max()

    class_map = {0: "down", 1: "sideways", 2: "up"}
    prediction = class_map.get(pred_class, "unknown")

    threshold_high_volume = 20000
    threshold_high_liquidity = 50000
    details = []
    if token_data["volume24h"] > threshold_high_volume:
        details.append("High trading volume detected—lots of buying and selling.")
    else:
        details.append("Low volume—market may be stagnant or illiquid.")
    if token_data["liquidityUsd"] > threshold_high_liquidity:
        details.append("Strong liquidity—less price manipulation risk.")
    else:
        details.append("Weak liquidity—price may be easily manipulated.")
    if token_data["price"] < 1e-5:
        details.append("Very low price—could be high risk or new token.")

    reasoning = (
        f"Based on live data → Price: {token_data['price']}, "
        f"Volume24h: {token_data['volume24h']} ({details[0]}), "
        f"LiquidityUsd: {token_data['liquidityUsd']} ({details[1]}). "
        f"{details[2] if len(details) > 2 else ''} "
        f"Model predicts '{prediction}' with confidence {round(prob,2)}"
    )
    return {"prediction": prediction, "confidence": round(prob,2), "reasoning": reasoning}

@app.get("/api")
def home():
    return {"message": "Welcome to Solana Memecoin Predictor powered by live ML!"}

@app.get("/api/predict")
def predict(symbol: str = Query(..., description="Token symbol or address")):
    token_data = fetch_token_data(symbol)
    if not token_data:
        return {"error": f"Token '{symbol}' not found on Solana."}
    result = predict_trend(token_data)
    return {**token_data, **result}

@app.get("/api/latest-tokens")
def latest_tokens():
    try:
        resp = requests.get("https://api.dexscreener.com/latest/dex/tokens/solana")
        data = resp.json()
        pairs = data.get("pairs")
        tokens = []
        if pairs is not None:
            for item in pairs:
                tokens.append({
                    "symbol": item["baseToken"]["symbol"],
                    "name": item["baseToken"]["name"],
                    "price": float(item.get("priceUsd", 0))
                })
        if tokens:
            return {"memecoins": tokens[:8]}
        else:
            raise Exception("No trending tokens in API.")
    except Exception as e:
        print(f"Error fetching trending tokens: {e}")
        # ONLY SHOW NAMES, NOT ADDRESSES IN THE UI!
        static_tokens = [
            {"symbol": "$TROLL", "name": "$TROLL", "price": 0.00001},
            {"symbol": "$SHITCOIN", "name": "$SHITCOIN", "price": 0.00002},
            {"symbol": "$NUB", "name": "$NUB", "price": 0.00003},
            {"symbol": "$WIF", "name": "$WIF", "price": 0.00004}
        ]
        return {"memecoins": static_tokens}

@app.get("/api/history")
def price_history(symbol: str = Query(..., description="Token symbol or address")):
    try:
        url = f"https://api.dexscreener.com/latest/dex/chart/{symbol}?network=solana&interval=5m"
        resp = requests.get(url)
        if not resp.ok or not resp.text:
            return {"history": [], "future": []}
        data = resp.json()
        history = []
        if "points" in data and len(data["points"]) > 10:
            history = [{"time": p["t"], "price": p["c"]} for p in data["points"] if "t" in p and "c" in p]
        future = []
        if history:
            last_time = history[-1]["time"]
            last_price = history[-1]["price"]
            for i in range(1, 13):
                drift = random.uniform(-0.01, 0.03)
                last_price = last_price * (1 + drift)
                future.append({"time": last_time + i * 5 * 60 * 1000, "price": round(last_price, 8)})
        return {"history": history, "future": future}
    except Exception as e:
        print(f"Error fetching history for {symbol}: {e}")
        return {"history": [], "future": []}

@app.get("/api/token-info")
def token_info(symbol: str = Query(..., description="Token symbol or address")):
    data = fetch_token_data(symbol)
    if not data:
        return {"error": f"Token '{symbol}' not found on Solana."}
    return data

app.mount("/", StaticFiles(directory="static", html=True), name="static")
