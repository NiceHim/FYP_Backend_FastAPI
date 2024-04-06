import pymongo
import os
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
from DRQN_GRU import DRQN_GRU
from datetime import datetime
from polygon import RESTClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pickle import load
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Ticker(BaseModel):
    ticker: str


device = torch.device("cpu")
robust_scaler = load(open("./model/robust_scaler.pkl", "rb"))
minmax_scaler = load(open("./model/minmax_scaler.pkl", "rb"))
standard_scaler = load(open("./model/standard_scaler.pkl", "rb"))
polygon_client = RESTClient(os.getenv("POLYGON_IO_API_KEY"))
mongoDB_client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
db = mongoDB_client[os.getenv("MONGODB_DATABASE_NAME")]


@app.post("/tradeSignal/", status_code=201)
async def getTradeSignal(body: Ticker):
    ticker = body.ticker
    pipeline = [
        {
            "$match": {
                "ticker": ticker 
            }
        },
        {
            "$sort": {
                "timestamp": -1
            }
        },
        {
            "$limit": 199
        },
        {
            "$sort": {
                "timestamp": 1
            }
        }
    ]
    data = list(db["forex"].aggregate(pipeline))
    df = pd.DataFrame(data)
    
    aggs = polygon_client.get_previous_close_agg(f"C:{ticker}")
    previous_close_datetime = datetime.fromtimestamp(round(aggs[0].timestamp / 1000))
    previous_close = [{
        "ticker": aggs[0].ticker.replace("C:", ""),
        "timestamp": pd.Timestamp(previous_close_datetime),
        "open": aggs[0].open,
        "close": aggs[0].close,
        "high": aggs[0].high,
        "low": aggs[0].low,
        "volume": aggs[0].volume,
        "VWAP": aggs[0].vwap
    }]

    previous_close_df = pd.DataFrame(previous_close)
    
    temp_df = pd.concat([df[["ticker", "timestamp", "open", "close", "high", "low", "volume", "VWAP"]], previous_close_df], ignore_index=True)
    temp_df.ta.hma(length=14, append=True)
    temp_df.ta.sma(length=10, append=True)
    temp_df.ta.sma(length=20, append=True)
    temp_df.ta.sma(length=50, append=True)
    temp_df.ta.sma(length=100, append=True)
    temp_df.ta.sma(length=200, append=True)
    temp_df.ta.log_return(cumulative=False, append=True)
    temp_df.ta.rsi(append=True)
    temp_df.ta.mfi(append=True)
    temp_df.ta.natr(append=True)
    temp_df.dropna(inplace=True)
    temp_df.reset_index(inplace=True)
    temp_df.drop(columns=["index"], inplace=True)

    db["forex"].insert_many(temp_df.to_dict("records"))
    df = df.iloc[-(int(os.getenv("WINDOW_SIZE"))-1):]
    df = pd.concat((df, temp_df), ignore_index=True)
    df.drop(columns=["_id", "ticker", "timestamp", "volume"], inplace=True)

    data_robust_scaler = robust_scaler.transform(df[["open", "close", "high", "low", "VWAP", "HMA_14","SMA_10", "SMA_20", "SMA_50", "SMA_100", "SMA_200", "NATR_14"]].values)
    data_minmax_scaler = minmax_scaler.transform(df[["RSI_14", "MFI_14"]].values)
    data_standard_scaler = standard_scaler.transform(df[["LOGRET_1"]].values)
    
    final_data = np.concatenate((data_robust_scaler, data_minmax_scaler, data_standard_scaler), axis=1)
    DRQN_GRU_model = DRQN_GRU(input_size=final_data.shape[1], output_size=2).to(device=device)
    DRQN_GRU_model.load_state_dict(torch.load(f"./model/Best_{ticker}_DRQN_GRU_policy_net.pt", map_location=device))
    DRQN_GRU_model.eval()
    input_data = torch.tensor(final_data, dtype=torch.float, device=device).unsqueeze(0)
    output = DRQN_GRU_model(input_data)
    action = torch.argmax(output).detach().cpu().numpy()
    action_str = "buy" if action == 0 else "sell"
    return {
        "message": {
            "action": action_str,
            "previousClosePrice": aggs[0].close,
            "previousCloseTimestamp": aggs[0].timestamp
        }
    }

@app.get("/")
async def root():
    return {"message": "Hello World!"}