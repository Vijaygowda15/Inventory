from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/Vijay/OneDrive/Desktop/Inventory/Inventory.csv")

# Prepare features and target
X = df.drop(columns=['Item_ID', 'Item_Name', 'Stock_Quantity'])
y = df['Stock_Quantity']
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train TPOT model
tpot = TPOTRegressor(verbosity=2, generations=5, population_size=20, random_state=42)
tpot.fit(X_train, y_train)

# Predictions and evaluation
y_pred = tpot.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared score: {r2}")

# Export the pipeline
tpot.export('tpot_pipeline.py')
