# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:14:17 2023

@author: PC
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

# Datos de ejemplo
data = {
    'feliz': ['Coco', 'La La Land', 'Shrek', 'Up'],
    'triste': ['Titanic', 'Bajo el sol de la Toscana', 'Marley & Me'],
    'emocionado': ['El señor de los anillos', 'Star Wars', 'Vengadores: Endgame'],
    'relajado': ['El gran hotel Budapest', 'Lost in Translation', 'El viaje de Chihiro'],
    'asustado': ['El exorcista', 'El resplandor', 'It'],
    'miedo': ['Insidious', 'El conjuro', 'Annabelle'],
    'confundido': ['Donnie Darko', 'Matrix', 'Origen']
}

# Preprocesamiento de datos
texts = []
labels = []
for label, movies in data.items():
    texts.extend([f"{label} {movie}" for movie in movies])
    labels.extend([label] * len(movies))

# Tokenización
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

# Secuencias de entrada
sequences = tokenizer.texts_to_sequences(texts)
max_sequence_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

# One-hot encoding de las etiquetas
label_dict = {'feliz': 0, 'triste': 1, 'emocionado': 2, 'relajado': 3, 'asustado': 4, 'miedo': 5, 'confundido': 6}
labels = [label_dict[label] for label in labels]
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_dict))

# Definición del modelo
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_len))
model.add(LSTM(100))
model.add(Dense(len(label_dict), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, one_hot_labels, epochs=50, verbose=2)

# Función para predecir el estado de ánimo y sugerir una película
def sugerir_pelicula_deep_learning(entrada_usuario):
    input_sequence = tokenizer.texts_to_sequences([entrada_usuario])
    padded_input = pad_sequences(input_sequence, maxlen=max_sequence_len, padding='post')
    predicted_label = np.argmax(model.predict(padded_input), axis=-1)[0]

    estado_usuario = [key for key, value in label_dict.items() if value == predicted_label][0]
    peliculas_estado = data[estado_usuario]
    sugerencia = random.choice(peliculas_estado)

    return f"Te sugiero ver '{sugerencia}' para un estado de ánimo {estado_usuario}."

# Ejemplo de uso
estado_usuario = input("¿Cuál es tu estado de ánimo actual? (feliz, triste, emocionado, relajado, asustado, miedo, confundido): ")
resultado = sugerir_pelicula_deep_learning(estado_usuario)
print(resultado)