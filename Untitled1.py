#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset
data = pd.read_excel(r"C:\Users\mrng shift\Desktop\New_flavour_data.xlsx")

# Tokenize the taste descriptions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Taste Description'])
vocab_size = len(tokenizer.word_index) + 1

# Encode the target descriptions (y values)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Taste Description'])

# Prepare sequences of molecule combinations
max_sequence_length = 5  # Set an appropriate maximum sequence length
X = data[[' Molecules A', 'Molecules B']].values
X_encoded = [tokenizer.texts_to_sequences([f"{m1}, {m2}"])[0] for m1, m2 in X]
X_padded = pad_sequences(X_encoded, maxlen=max_sequence_length)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# Define a Bidirectional LSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=47, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(47)))  # Increased complexity
model.add(Dense(vocab_size, activation='sigmoid'))  # Output dimension matches vocab_size

# Compile the model with categorical cross-entropy loss
model.compile(loss="categorical_crossentropy", optimizer='RMSprop', metrics=['accuracy'])

model.summary()

# Train the model with more epochs
model.fit(X_train, tf.keras.utils.to_categorical(y_train, num_classes=vocab_size), epochs=2, batch_size=64, validation_data=(X_test, tf.keras.utils.to_categorical(y_test, num_classes=vocab_size)))

# Predict taste descriptions
def predict_taste(molecule1, molecule2):
    sequence = tokenizer.texts_to_sequences([f"{molecule1}, {molecule2}"])[0]
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length)
    predicted_sequence = model.predict(padded_sequence)
    predicted_word_index = np.argmax(predicted_sequence)
    predicted_word = label_encoder.inverse_transform([predicted_word_index])[0]
    return predicted_word

# Example usage
molecule1 = 'Terpineol'
molecule2 = 'Borneol'
predicted_taste = predict_taste(molecule1, molecule2)
print(f"Predicted Taste Description: {predicted_taste}")

