import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def build_and_train_model(X_train, y_train, X_val, y_val):
    # Build the model
    model = Sequential([
        Dense(32, activation='relu', input_dim=8),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val))
    
    return model
