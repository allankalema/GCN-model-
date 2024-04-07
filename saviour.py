

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical

# pip install shap

import pandas as pd

# Load the dataset
url = 'heart_statlog_cleveland_hungary_final.csv'
df = pd.read_csv(url)

# Split the dataset into features (inputs) and target (output) variables
X = df.drop(columns=['target'])  # Features
y = df['target']  # Target variable

# Print the shape of the features and target variables
print("Shape of features (X):", X.shape)
print("Shape of target variable (y):", y.shape)

from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the input features
X_normalized = scaler.fit_transform(X)

# Convert the normalized features back to a DataFrame
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

# Print the first few rows of the normalized features
print("Normalized Features:")
print(X_normalized_df.head())

from sklearn.model_selection import train_test_split

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print the shapes of the resulting datasets
print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of y_test:", y_test.shape)

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class GraphConvLayer(Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=(input_shape[1], self.units))
        self.bias = self.add_weight("bias", shape=(self.units,))
        super(GraphConvLayer, self).build(input_shape)

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.kernel)
        outputs = tf.add(outputs, self.bias)
        outputs = self.activation(outputs)
        return outputs

# Define GCN model
inputs = tf.keras.Input(shape=(X_train.shape[1],))
x = GraphConvLayer(32)(inputs)
x = GraphConvLayer(64)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

import shap

# Create SHAP explainer
explainer = shap.Explainer(model, X_train)

# Compute SHAP values
shap_values = explainer.shap_values(X_test)

# Visualize feature importance
shap.summary_plot(shap_values, X_test)

from sklearn.metrics import classification_report, confusion_matrix

# Predict labels for the test set
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")

# Compute classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No Heart Disease", "Heart Disease"],
            yticklabels=["No Heart Disease", "Heart Disease"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

"""Explainability"""



# Loop through each random row
for i, row in enumerate(random_rows.iterrows()):
    index, data = row
    # Get model prediction for the current row
    prediction_prob = model.predict(data.values.reshape(1, -1))[0]
    prediction = 1 if prediction_prob > 0.5 else 0  # Apply threshold of 0.5

    # Compute SHAP values for the current row
    shap_values = explainer.shap_values(data.values.reshape(1, -1))

    # Print model prediction and ground truth label
    print(f"Row {i + 1} - Prediction: {prediction}")

    # Visualize feature contributions for the current row
    shap.force_plot(0.5, shap_values, data, matplotlib=True)  # Assuming the base value is 0.5

# # Assuming 'model' is your trained TensorFlow model
# model.save('my_model.h5')

























