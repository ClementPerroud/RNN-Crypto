"""
API LIMITS
1200 requests per minute
10 orders per second
100,000 orders per 24hrs


"""


import asyncio
import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from binance.client import Client
import time, datetime
import tensorflow as tf
from common_neural_network_functions import *


CLOCK_TICK = 60 #seconds



client = Client(api_key, api_secret)



MAIN_DF= pd.DataFrame()
training_dataset = gather_dataset()
model = tf.keras.models.load_model(MODEL_PATH)

for crypto in CRYPTOS:
    crypto_backup = crypto
    crypto = crypto + "USDT"
    #On commence par récupérer les datas sur les 60 dernières minutes
    untilThisDate = datetime.datetime.now()
    sinceThisDate = untilThisDate - datetime.timedelta(hours = 2)
    candle = client.get_historical_klines(crypto, Client.KLINE_INTERVAL_1MINUTE, str(sinceThisDate), str(untilThisDate))
    df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
    df.dateTime = list(map(lambda x: int(x / 1000), df.dateTime))

    df= df[["dateTime", "close", "volume"]]
    df.rename(columns={"close": f"{crypto_backup}-USDT_close", "volume": f"{crypto_backup}-USDT_volume"}, inplace=True)
    df.set_index("dateTime", inplace= True)

    if len(MAIN_DF) == 0:
        MAIN_DF= df
    else:
        MAIN_DF = MAIN_DF.join(df)


TEMP_DF = { crypto + "-USDT" : [] for crypto in CRYPTOS}



async def crypto_stream(client, bm, symbol_crypto):
    global TEMP_DF
    #On lance l'écoute en temps réel
    async with bm.kline_socket(symbol_crypto) as stream:
        while True:
            ts1 = time.time()
            res= await asyncio.gather(
                stream.recv(),
                asyncio.sleep(CLOCK_TICK)
            )
            res = res[0]
            new_df = [res["E"], res["k"]["c"], res["k"]["c"]]#, columns=["datetime", f"{symbol_crypto}_close", f"{symbol_crypto}_volume"])
            #new_df.set_index("datetime", inplace= True)

            TEMP_DF[symbol_crypto.replace("USDT", "-USDT")].append(new_df)
            if handleArrivingDatas():#La fonction vérifie si on peut rajouter une ligne de data
                #Si oui, on peut passe le nouveau set d'input à l'IA
                df_for_IA = preprocess_input_for_prediction(training_dataset, MAIN_DF)
                print(df_for_IA)
                inputs_for_IA = df_for_IA.values.tolist()
                print(inputs_for_IA)
                prediction = model.predict([inputs_for_IA])
                print(prediction)



def handleArrivingDatas():
    global MAIN_DF
    global TEMP_DF
    verif = True
    for crypto in CRYPTOS:
        if len(TEMP_DF[crypto + "-USDT"]) == 0:
            verif = False

    if verif:
        row_to_add = []
        for crypto in CRYPTOS:
            temp_array = TEMP_DF[crypto + "-USDT"].pop() #drop(index = TEMP_DF[crypto + "USDT"].index.values[0], inplace=True)

            if len(row_to_add) == 0:
                row_to_add = temp_array
            else:
                row_to_add = row_to_add + temp_array[1:]

        row_to_add = list(map(float, row_to_add))
        df_to_add = pd.DataFrame([row_to_add[1:]], columns= MAIN_DF.columns, index= [int(row_to_add[0]/10e3)])

        MAIN_DF = MAIN_DF.append(df_to_add, ignore_index=False)

        MAIN_DF.drop(index = MAIN_DF.index.values[0], inplace= True)
        return True
    return False
        #MAINTENANT IL FAUT EXPLOITER LES DATAS !!!


async def main():
    async_client = await AsyncClient.create(api_key, api_secret)
    profils = await async_client.get_exchange_info()

    bm = BinanceSocketManager(async_client)



    #BEGIN TO STREAM OUR COIN

    await asyncio.gather(
        *[crypto_stream(async_client, bm, CRYPTOS[i] + "USDT") for i in range(len(CRYPTOS))]
    )


    await async_client.close_connection()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())