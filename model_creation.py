import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv("Training.csv")

# Features and target
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build FeedForward neural network
model = Sequential([
    Dense(128, input_shape=(X.shape[1],), activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Add EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save model and encoder
model.save("disease_prediction_enhanced.h5")
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


import numpy as np
from sklearn.metrics import classification_report

# Predicting on test set
y_pred_probs = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Converting numeric predictions to disease names
predicted_diseases = label_encoder.inverse_transform(y_pred_labels)
true_diseases = label_encoder.inverse_transform(y_true_labels)

# Show a few predictions
for i in range(10):
    print(f"Actual: {true_diseases[i]}  |  Predicted: {predicted_diseases[i]}")

# Optional: classification report
print("\nClassification Report:\n")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))


import matplotlib.pyplot as plt

# Train model and capture history
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()