import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.utils import to_categorical

# Initialize tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(["This movie is awesome.", "This movie is terrible."])

# Preprocess data
sequences = tokenizer.texts_to_sequences(["This movie is awesome.", "This movie is terrible."])
padded_sequences = pad_sequences(sequences, maxlen=80, truncating='post', padding='post')
labels = [1, 0] # 1 is positive, 0 is negative
labels = to_categorical(labels)

# Initialize model
model = Sequential()
model.add(Embedding(5000, 128, input_length=80))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(padded_sequences, labels, epochs=5, verbose=1)

# Get user input
user_input = input("Enter your movie review: ")
user_input = tokenizer.texts_to_sequences([user_input])
user_input = pad_sequences(user_input, maxlen=80, truncating='post', padding='post')

# Make prediction
prediction = model.predict(user_input)

# Print result
if prediction[0][0] > prediction[0][1]:
    print("Positive sentiment.")
else:
    print("Negative sentiment.")
    
    
    
    