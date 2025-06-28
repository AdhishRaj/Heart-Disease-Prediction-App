import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

data = pd.read_csv("heart_1.csv")

selected_features = ['cp', 'sex', 'thalach', 'exang', 'oldpeak']
x = data[selected_features]
y = data['target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x, y)

dump(model, 'random_forest_model.joblib')