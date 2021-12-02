import time

#TUTO : Comment ça fonctionne ?

#Etape 1 : Choisir les cryptos sur lesquels on veut travailler

CRYPTOS = ["BTC","BNB", "ETH", "LUNA", "BCH", "LTC"]

#Etape 2 : Collecter les datas
#Choisir la durée de sur laquelle on récolte les datas via la varaible

DURATION = 300 #days (interval d'une minute)

#Etape 3 : Entrainer le modèle
#Définir le fichier dans le quel est stocké les datas récoltés précédemment
DATASET_FILE = "2021.02.04-2021.12.01-300DAYS-BTC_BNB_ETH_LUNA_BCH_LTC"
#Définir les paramètres de l'entrainement
SEQ_LEN = 60#minutes
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "BTC-USDT"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

#Etape 4 : Lancer le modèle en simulation
# Récupérer le nouveau model et entrer son path ci-dessous
MODEL_PATH = "LSTM-BTC-USDT_on_BNB-BTC-PSG.model"






#GENERAL VARIABLES
api_key = "XkU7Ct0YL3RDgB3muuoxVu5CnhoSj0vNarj9CBYj0fLMhxicaqx018adS2Lzv7GJ"
api_secret = "AoMhLtvRb94X7Lf2TEKvfPOusuWC2rVW9oeGPE2h48ICDaQ3dN3SibZ08Y8o5yf4"



#HARVEST VARIABLES



#TRAINING VARIABLES
