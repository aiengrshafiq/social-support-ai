# ml/train_dummy_model.py
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# --- Create Simple Synthetic Data ---
# In reality, this would come from historical application data
data = {
    'income': [2000, 3000, 5000, 1500, 8000, 12000, 4000, 6000, 2500, 7000],
    'family_size': [1, 2, 4, 3, 1, 5, 2, 3, 1, 4],
    # Add other relevant features based on ValidatedApplicationData
    # e.g., 'employment_duration_months': [6, 12, 24, 3, 36, 60, 18, 20, 9, 48],
    'eligible': [0, 0, 1, 0, 1, 1, 1, 1, 0, 1] # 1 = Approved, 0 = Declined (Target variable)
}
df = pd.DataFrame(data)

# --- Simple Feature Engineering (if needed) ---
# Example: Income per capita
# df['income_per_capita'] = df['income'] / df['family_size']

# --- Select Features and Target ---
features = ['income', 'family_size'] # Adjust as needed
target = 'eligible'
X = df[features]
y = df[target]

# --- Train/Test Split (Optional for dummy) ---
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
# Using HistGradientBoostingClassifier as decided earlier
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X, y) # Train on the full tiny dataset for the dummy

# --- Save Model ---
model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'eligibility_model_v1.joblib')
joblib.dump(model, model_path)

print(f"Dummy model trained and saved to: {model_path}")
print(f"Input features expected: {features}")