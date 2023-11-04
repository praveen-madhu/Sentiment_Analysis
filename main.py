import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.utils import to_categorical


tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(["This movie is awesome.", "This movie is terrible."])


sequences = tokenizer.texts_to_sequences(["This movie is awesome.", "This movie is terrible."])
padded_sequences = pad_sequences(sequences, maxlen=80, truncating='post', padding='post')
labels = [1, 0]  
labels = to_categorical(labels)


model = Sequential()
model.add(Embedding(5000, 128, input_length=80))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(2, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(padded_sequences, labels, epochs=5, verbose=1)

st.title("Movie Sentiment Analysis")


user_review = st.text_input("Enter your movie review:")

if user_review:
    
    user_input = tokenizer.texts_to_sequences([user_review])
    user_input = pad_sequences(user_input, maxlen=80, truncating='post', padding='post')

    
    prediction = model.predict(user_input)

    
    if prediction[0][0] > prediction[0][1]:
        st.write("Prediction: Positive sentiment.")
    else:
        st.write("Prediction: Negative sentiment.")
