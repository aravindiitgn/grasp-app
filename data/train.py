import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the dataset
csv_path = 'final_workspace_volumes.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_path)

# One-Hot Encode the Object Shapes
def one_hot_encode_shape(shape_column):
    """Converts the shape list (e.g., [0, 1, 0]) to individual one-hot encoded columns."""
    return pd.DataFrame(shape_column.apply(eval).to_list(), columns=['Cuboid', 'Sphere', 'Cylinder'])

# One-hot encode Object 1 and Object 2 shapes
df_obj1_shape = one_hot_encode_shape(df['Object 1 Shape (Cuboid, Sphere, Cylinder)'])
df_obj2_shape = one_hot_encode_shape(df['Object 2 Shape (Cuboid, Sphere, Cylinder)'])

# Combine one-hot encoded features with the original DataFrame
df = pd.concat([df, df_obj1_shape, df_obj2_shape], axis=1)

# Drop the original shape columns
df = df.drop(columns=['Object 1 Shape (Cuboid, Sphere, Cylinder)', 'Object 2 Shape (Cuboid, Sphere, Cylinder)'])

# Define input features (X) and target variables (y)
X = df.drop(columns=['Remaining Workspace Volume after first grasp', 'Remaining Workspace Volume after both grasps'])
y = df[['Remaining Workspace Volume after first grasp', 'Remaining Workspace Volume after both grasps']]

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
scaler_path = 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved as {scaler_path}")

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model using a Gradient Boosting Regressor for both targets
model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
print(y_pred)
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='variance_weighted')

print(f"MAE (Volume after 1st grasp, Volume after 2nd grasp): {mae}")
print(f"MSE (Volume after 1st grasp, Volume after 2nd grasp): {mse}")
print(f"R² Score: {r2}")

# Save the model for future use
joblib.dump(model, 'final_workspace_volume_model_new.pkl')
print("Model saved as final_workspace_volume_model.pkl")
import matplotlib.pyplot as plt

importances = model.estimators_[0].feature_importances_
feature_names = X.columns
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
