import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
import os
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
import keras_tuner as kt
from keras import ops
#if not os.path.exists('Mod16-32-16.keras'):
#def build_model(hp):
inputs = keras.Input(shape=(10,))
layer1 = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005))
layer2 = layers.Dropout(0.2)
layer3 = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005))
layerOut = layers.Dense(1, activation='sigmoid')
output = layerOut(layer3(layer2(layer1(inputs))))
#output = layerOut(layer1(inputs))
model = keras.Model(inputs=inputs, outputs=output, name = 'Mod16-32-16')
keras.saving.save_model(model, "Mod16-32-16.keras")
del model

model = keras.models.load_model("Mod16-32-16.keras")
model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

df = pd.read_csv("data.csv")
sample_df = df.sample(frac=0.3, random_state=42)
remaining_df = df.drop(sample_df.index)
sample_df.to_csv("val.csv", index=False)
remaining_df.to_csv("train.csv", index=False)

xData = np.loadtxt("train.csv", delimiter=",", skiprows=1)
yData = xData[:,0]
xData = xData[:, 1:]
xEval = np.loadtxt("val.csv", delimiter=",", skiprows=1)
yEval = xEval[:,0]
xEval = xEval[:, 1:]

sm = SMOTE(random_state=42)
xData, yData = sm.fit_resample(xData, yData)

scaler = MinMaxScaler()
scaler = scaler.fit(xData)
xData = scaler.transform(xData)
xEval = scaler.transform(xEval)

history = model.fit(xData, yData, validation_data=(xEval, yEval), batch_size=32, epochs=200, verbose=1, callbacks=[keras.src.callbacks.EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True), keras.src.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=15,min_lr=1e-6)])
evals = model.evaluate(xEval, yEval, verbose=1)
print("Loss, Accuracy: ", evals)
model.save("Mod16-32-16.keras")
del model

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()