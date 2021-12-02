from sklearn import preprocessing
from collections import deque
import numpy as np
import random
from env import *
import pandas as pd
import glob
import os

def classify(current, future):
    if float(future) > float(current):
        return 1
    return 0
def preprocess_input_for_prediction(training_df, df): #L'objectif ici et d'avoir un preprocess simililaire
    training_df = training_df.append(df)
    for col in training_df.columns:
        if col != "target":
            df = df.replace(0, np.nan)
            training_df.dropna(inplace=True)
            training_df[col] = training_df[col].astype(float)
            training_df[col]= training_df[col].pct_change() # on transforme les valeurs en % de variations
            training_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            training_df.dropna(inplace= True) #on vire les NaN et les Infiny (Not a Number)

            training_df[col] = preprocessing.scale(training_df[col].values) #on normalise les datas de 0 à 1 (trouver un meilleur algo plus tard)
    df = training_df.drop(index= training_df.index.values[0:-60])
    return df
def preprocess_df(df):
    df.drop(columns ='future', inplace = True)
    for col in df.columns:
        if col != "target":

            df[col]= df[col].pct_change() # on transforme les valeurs en % de variations
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace= True) #on vire les NaN et les Infiny (Not a Number)

            df[col] = preprocessing.scale(df[col].values) #on normalise les datas de 0 à 1 (trouver un meilleur algo plus tard)


    df.dropna(inplace = True)
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
            #on stocke dans chaque ligne une sequence contenant, la dernière minute des cours des actions et le résultat attendu

    random.shuffle(sequential_data) # on randomise l'ordre de ses sequences pour avoir de meilleurs résultats

    # OBJECTIF : avoir le meme nombre de résultat ou nous devons vendre qu'acheter pour ne pas biaiser l'algorythme
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    Y = []

    for seq, target in sequential_data:
        X.append(seq)
        Y.append(target)
    return np.array(X).astype("float32"), np.array(Y)

def gather_dataset(**kwargs):
    max_row = kwargs.get("max_row", 0)
    main_df = pd.DataFrame()
    datas_titles = ["time", "low", "high","open", "close", "volume"]

    datas_path = glob.glob(f"training-crypto_datas/{DATASET_FILE}/*.csv")
    for data_path in datas_path:
        ratio = os.path.basename(data_path).split(".csv")[0]
        dataset_path = data_path
        if max_row > 0:
            df = pd.read_csv(dataset_path, names=datas_titles, nrows = max_row)
        else:
            df = pd.read_csv(dataset_path, names=datas_titles)
        df.rename(columns={"close":f"{ratio}_close", "volume":f"{ratio}_volume"}, inplace= True)
        df.set_index("time", inplace= True)
        df = df[[f"{ratio}_close", f"{ratio}_volume"]]

        if len(main_df) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)
    return main_df