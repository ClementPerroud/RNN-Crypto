""""
Idée 1 : changer les inputs avec des datas plus éloignés
- 45 dernières minutes
- 5 dernières heures
- 5 derniers jours
- 5 dernières semaines

Idée 2 : ajouter une fonction logarithmique sur les inputs

"""
#!pip install -q -U keras-tuner

import collections
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, LSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from common_neural_network_functions import *
from keras_tuner import RandomSearch, Hyperband
from tensorflow.keras import losses, optimizers
from keras_tuner.engine.hyperparameters import HyperParameters

LOG_DIR = "2021.12.01"
main_df = gather_dataset(max_row=100000)
print(main_df)

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

def build_model_test(hp):
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

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    opt = tf.keras.optimizers.Adam(learning_rate= hp_learning_rate, decay = 1e-6)

    model.compile(loss= "sparse_categorical_crossentropy", optimizer= opt, metrics = ["accuracy"])

    tensorboard = TensorBoard(log_dir= f"logs/{NAME}")


    #checkpoint_filepath = "models/RNN_Final-{epoch:02d}-{val_accuracy:.3f}.hd5"
    #checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    #history = model.fit(train_x, train_y,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(validation_x, validation_y), callbacks=[tensorboard, checkpoint])
    #model.save(f'LSTM-{RATIO_TO_PREDICT}_on_{"-".join(CRYPTOS)}.model')

    return model

def build_model(hp):
    model = Sequential()

    model.add(LSTM(hp.Int("input_units",32, 256, 32), input_shape = (train_x.shape[1:]), return_sequences= True))
    #model.add(Dropout(0.2))
    model.add(BatchNormalization())

    for i in range(hp.Int("n_layers", 1, 4)):
      model.add(LSTM(hp.Int(f"lstm_{i}_units",32, 256, 32), input_shape = (train_x.shape[1:]), return_sequences= True))
      #model.add(Dropout(0.1))
      model.add(BatchNormalization())


    model.add(Dense(hp.Int("prev_final_layers", 16, 128,16), activation= "relu"))
    #model.add(Dropout(0.2))

    #model.add(Dense(hp.Int("prev_final_layers", 2, 8,2), activation = "softmax"))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile( loss=losses.CategoricalCrossentropy(from_logits=False), optimizer= optimizers.Adam(learning_rate = hp_learning_rate), metrics = ["accuracy"])

    tensorboard = TensorBoard(log_dir= f"logs/{NAME}")


    #checkpoint_filepath = "models/RNN_Final-{epoch:02d}-{val_accuracy:.3f}.hd5"
    #checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    #history = model.fit(train_x, train_y,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(validation_x, validation_y), callbacks=[tensorboard, checkpoint])
    #model.save(f'LSTM-{RATIO_TO_PREDICT}_on_{"-".join(CRYPTOS)}.model')

    return model
#model = build_model(" ")
#model.fit(train_x, train_y,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(validation_x, validation_y), callbacks=[tensorboard, checkpoint])
tuner = Hyperband(
    build_model_test,
    objective="val_accuracy",
    max_epochs = 15,
    factor = 3,
    #executions_per_trial = 1,
    directory = LOG_DIR,
    overwrite=True
)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x= train_x,
             y= train_y,
             epochs= 5,
             batch_size= 64,
             validation_data= (validation_x, validation_y))

print(np.shape(train_x))
print(np.shape(train_y))

tuner.search([train_x], [train_y], epochs=50, validation_split=0.2, callbacks=[stop_early])
# Get the optimal hyperparameters
#best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
"""
print(f'''
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
''')
"""