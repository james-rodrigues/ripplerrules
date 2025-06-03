import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
from imodels import RuleFitClassifier  # From the imodels library

# Step 1: Create synthetic dispute data
np.random.seed(42)
n = 1000
df = pd.DataFrame(
    {
        "dispute_amount": np.random.randint(100, 10000, n),
        "dispute_type": np.random.choice(
            ["fraud", "product_quality", "billing_error"], n
        ),
        "customer_age": np.random.randint(18, 70, n),
        "account_tenure": np.random.randint(1, 15, n),
        "resolved_within_7_days": np.random.choice([0, 1], n, p=[0.3, 0.7]),
    }
)

# Step 2: Encode categorical features
le = LabelEncoder()
df["dispute_type_encoded"] = le.fit_transform(df["dispute_type"])

# Features and label
X = df[["dispute_amount", "dispute_type_encoded", "customer_age", "account_tenure"]]
y = df["resolved_within_7_days"]

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))

# Step 5: RuleFit model
rf_model = RuleFitClassifier()
rf_model.fit(X_train.values, y_train.values, feature_names=X.columns.tolist())
rf_preds = rf_model.predict(X_test)
print("RuleFit Accuracy:", accuracy_score(y_test, rf_preds))

# Step 6: View extracted rules
import pandas as pd

# Create a DataFrame with rules and their coefficients
rules_data = []
for i, rule in enumerate(rf_model.rules_):
    if i < len(rf_model.coef) and rf_model.coef[i] != 0:
        rules_data.append({
            'rule': str(rule),
            'coef': rf_model.coef[i],
            'abs_coef': abs(rf_model.coef[i])
        })

rules_df = pd.DataFrame(rules_data)

# Sort by absolute coefficient value (importance)
if not rules_df.empty:
    rules_df = rules_df.sort_values(by="abs_coef", ascending=False)
    
    # Display top rules
    print("\nTop Rules:")
    print(rules_df[["rule", "coef", "abs_coef"]].head(10))
else:
    print("\nNo rules with non-zero coefficients found.")
