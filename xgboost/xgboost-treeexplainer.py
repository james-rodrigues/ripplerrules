import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Dispute Data
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'dispute_amount': np.random.randint(100, 10000, n),
    'dispute_type': np.random.choice(['fraud', 'product_quality', 'billing_error'], n),
    'customer_age': np.random.randint(18, 70, n),
    'account_tenure': np.random.randint(1, 15, n),
    'resolved_within_7_days': np.random.choice([0, 1], n, p=[0.3, 0.7])
})

# Encode categorical features
le = LabelEncoder()
df['dispute_type_encoded'] = le.fit_transform(df['dispute_type'])

# Prepare features and target
X = df[['dispute_amount', 'dispute_type_encoded', 'customer_age', 'account_tenure']]
y = df['resolved_within_7_days']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train XGBoost Classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Step 3: Explain model predictions with SHAP TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Step 4: Visualizations
# Summary plot (feature importance + direction)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# More detailed summary
shap.summary_plot(shap_values, X_test)

# Step 5: Force Plot for a Single Prediction
# Choose one example
i = 10
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[i], X_test.iloc[i])

