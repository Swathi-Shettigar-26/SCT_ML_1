# model.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# 1. Load Dataset
# ------------------------------
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print(train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].head())


# ------------------------------
# 2. Select Features & Target
# ------------------------------
X = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]  # features
y = train_data['SalePrice']  # target

# ------------------------------
# 3. Split Data (for evaluation)
# ------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 4. Train Model
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# ------------------------------
# 5. Evaluate Model
# ------------------------------
y_pred = model.predict(X_valid)

print("\nEvaluation Metrics:")
print("Mean Squared Error (MSE):", mean_squared_error(y_valid, y_pred))
print("R2 Score:", r2_score(y_valid, y_pred))

# ------------------------------
# 6. Visualize Predictions
# ------------------------------
plt.scatter(y_valid, y_pred, color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# ------------------------------
# 7. Predict on Test Data
# ------------------------------
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
test_predictions = model.predict(X_test)

# Save predictions to CSV
output = pd.DataFrame({
    "Id": test_data["Id"],   # Kaggle test set has "Id" column
    "SalePrice": test_predictions
})
output.to_csv("predictions.csv", index=False)

print("\nPredictions saved to predictions.csv")
