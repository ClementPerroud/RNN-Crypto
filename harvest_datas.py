import os
from binance.client import Client
import pandas as pd
import datetime, time
import os
from env import *





client = Client(api_key, api_secret)

untilThisDate = datetime.datetime.now()
sinceThisDate = untilThisDate - datetime.timedelta(days=DURATION)

path = f"training-crypto_datas/{sinceThisDate.strftime('%Y.%m.%d')}-{untilThisDate.strftime('%Y.%m.%d')}-{DURATION}DAYS-{'_'.join(CRYPTOS)}"

try:
    os.mkdir(path)
except:
    print("File already exist - Replacing in progress")



for crypto in CRYPTOS:
    crypto_stream = crypto + "USDT"
    # Calculate the timestamps for the binance api function

    # Execute the query from binance - timestamps must be converted to strings !
    candle = client.get_historical_klines(crypto_stream, Client.KLINE_INTERVAL_1MINUTE, str(sinceThisDate), str(untilThisDate))

    # Create a dataframe to label all the columns returned by binance so we work with them later.
    df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
    # as timestamp is returned in ms, let us convert this back to proper timestamps.
    df.dateTime = list(map(lambda x: int(x / 1000), df.dateTime))

    # on doit arriver à ça ["time", "low", "high", "open", "close", "volume"]
    df= df[["dateTime", "low","high","open","close", "volume"]]
    df.set_index("dateTime", inplace= True)
    df.to_csv(path + "/" + crypto + "-USDT.csv", header = False)

