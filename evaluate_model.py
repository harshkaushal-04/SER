import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import classification_report, accuracy_score

# Load the test dataset
X_test = np.load("X_test.npy")  # Replace with the correct path to your test features
y_test = np.load("y_test.npy")  # Replace with the correct path to your test labels

# Load the trained model
model = load_model("best_model.keras")  # Use the correct filename

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted class indices

# Check the dimensions of y_test
if len(y_test.shape) == 1:  # If y_test is not one-hot encoded
    y_true = y_test  # Use directly
else:  # If y_test is one-hot encoded
    y_true = np.argmax(y_test, axis=1)  # Convert to label indices

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred_classes)
print(f"Model Accuracy on Test Data: {accuracy:.2f}")

# Print classification report
EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'fear']  # Define emotion labels
report = classification_report(y_true, y_pred_classes, target_names=EMOTIONS)
print("\nClassification Report:\n")
print(report)
