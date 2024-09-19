import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d') 

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

features = ['Data', 'Ultimo', 'Abertura', 'Maxima', 'Minima', 'Vol', 'Var'] 
targets = ['Abertura', 'Maxima', 'Minima', 'Vol', 'Var']  # Lista de colunas alvo

X = df[features]
y = df[targets] 

# --- Divisão em Treino e Teste ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Crie o modelo de rede neural ---

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[len(features)]), 
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=len(targets))  # Camada de saída com unidades igual ao número de alvos
])
# --- Compile o modelo ---

model.compile(loss='mse', optimizer='adam')

# --- Treine o modelo ---

model.fit(X_train, y_train, epochs=100, verbose=0)

# --- Salve o modelo como um arquivo .onnx ---

onnx_model_path = "stock_model.onnx" 
tf.keras.models.save_model(model, onnx_model_path, save_format='onnx')

print(f"Modelo TensorFlow salvo em: {onnx_model_path}")
