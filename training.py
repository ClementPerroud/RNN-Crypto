""""
Idée 1 : changer les inputs avec des datas plus éloignés
- 45 dernières minutes
- 5 dernières heures
- 5 derniers jours
- 5 dernières semaines

Idée 2 : ajouter une fonction logarithmique sur les inputs

"""



import collections
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, LSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from common_neural_network_functions import *


main_df = gather_dataset(max_row=100000)

main_df["future"] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df["target"] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))


main_df.dropna(inplace=True)


times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
trainY_counter = collections.Counter(train_y)
print(f"Dont buys: {trainY_counter[0]}, buys: {trainY_counter[1]}")
validationY_counter = collections.Counter(validation_y)
print(f"VALIDATION Dont buys: {validationY_counter[0]}, buys: {validationY_counter[1]}")


print("-------------")
print("VALIDATION X")
print(validation_x[0])
print("-------------")
print("VALIDATION Y")
print(validation_y[0])


model = Sequential()
model.add(LSTM(128, input_shape = (train_x.shape[1:]), return_sequences= True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape = (train_x.shape[1:]), return_sequences= True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape = (train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation= "relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation = "softmax"))

opt = tf.keras.optimizers.Adam(learning_rate= 0.001, decay = 1e-6)

model.compile(loss= "sparse_categorical_crossentropy", optimizer= opt, metrics = ["accuracy"])

tensorboard = TensorBoard(log_dir= f"logs/{NAME}")


checkpoint_filepath = "models/RNN_Final-{epoch:02d}-{val_accuracy:.3f}.hd5"
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(train_x, train_y,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(validation_x, validation_y), callbacks=[tensorboard, checkpoint])
model.save(f'LSTM-{RATIO_TO_PREDICT}_on_{"-".join(CRYPTOS)}.model')
