import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import wittgenstein as lw
import pickle
import os

# -----------------------------
# 1. Generate Synthetic Dispute Data
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'dispute_amount': np.random.randint(100, 10000, n),
    'dispute_type': np.random.choice(['fraud', 'product_quality', 'billing_error'], n),
    'customer_age': np.random.randint(18, 70, n),
    'account_tenure': np.random.randint(1, 15, n),
})

# 2. Create target variable: 'action' (accept or challenge)
df['action'] = np.where(
    ((df['dispute_type'] == 'billing_error') | (df['account_tenure'] > 10)) &
    (df['dispute_amount'] < 5000),
    'accept',
    'challenge'
)

# 3. Encode categorical features
le_dispute_type = LabelEncoder()
df['dispute_type_encoded'] = le_dispute_type.fit_transform(df['dispute_type'])

le_action = LabelEncoder()
df['action_encoded'] = le_action.fit_transform(df['action'])  # 0 = accept, 1 = challenge

# 4. Train/test split
features = ['dispute_amount', 'dispute_type_encoded', 'customer_age', 'account_tenure']
X = df[features]
y = df['action_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Train XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)

# -----------------------------
# 6. Train RIPPER using wittgenstein
df_train = df.iloc[X_train.index].copy()
df_test = df.iloc[X_test.index].copy()

ripper = lw.RIPPER()
ripper.fit(
    df_train[['dispute_amount', 'dispute_type', 'customer_age', 'account_tenure', 'action']],
    class_feat='action',
    pos_class='accept'
)

# Get predictions from RIPPER
ripper_preds_raw = ripper.predict(df_test)
y_true = df_test['action'].values

# Convert boolean predictions to string labels
# True means the positive class ('accept'), False means negative class ('challenge')
ripper_preds = np.where(ripper_preds_raw, 'accept', 'challenge')

# Debug: Print first few predictions to see format
print("\n🔍 Debug Info:")
print(f"First 5 true labels: {y_true[:5]}")
print(f"First 5 RIPPER raw predictions: {ripper_preds_raw[:5]}")
print(f"First 5 RIPPER predictions: {ripper_preds[:5]}")
print(f"Unique true labels: {np.unique(y_true)}")
print(f"Unique RIPPER predictions: {np.unique(ripper_preds)}")

# Calculate accuracy
ripper_acc = accuracy_score(y_true, ripper_preds)

# Save models
models_dir = 'saved_models'
os.makedirs(models_dir, exist_ok=True)

# Save XGBoost model
xgb_model.save_model(f'{models_dir}/xgboost_model.json')

# Save RIPPER model and encoders using pickle
with open(f'{models_dir}/ripper_model.pkl', 'wb') as f:
    pickle.dump(ripper, f)

with open(f'{models_dir}/label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'dispute_type': le_dispute_type,
        'action': le_action
    }, f)

print(f"✅ Models saved to {models_dir}/")

# -----------------------------
# 7. Rule Export Functions

def parse_rule_str(rule_str):
    # RIPPER rules from wittgenstein come in format like: [dispute_type=billing_error^dispute_amount=<1113.9]
    # We need to extract conditions and determine the action (pos_class='accept')
    rule_str = str(rule_str).strip()
    
    # Remove brackets
    if rule_str.startswith('[') and rule_str.endswith(']'):
        conditions = rule_str[1:-1]
    else:
        conditions = rule_str
    
    # For RIPPER, if the rule matches, it predicts the positive class ('accept')
    # Otherwise, it predicts the negative class ('challenge')
    action = 'accept'  # This is the pos_class we specified
    
    return conditions, action

def generate_drools(ruleset):
    drl = []
    for idx, rule in enumerate(ruleset.rules):
        condition_str, action = parse_rule_str(str(rule))
        if not condition_str or condition_str == "ELSE":
            continue
            
        rule_name = f"Rule_{idx + 1}"
        
        # Convert RIPPER conditions to Drools format
        # Replace ^ with && for AND operations
        conditions = condition_str.replace("^", " && ")
        
        # Handle different comparison operators
        conditions = conditions.replace("=", " == ")
        conditions = conditions.replace("<", " < ")
        conditions = conditions.replace(">", " > ")
        
        # Handle range conditions like "10.0-12.0"
        import re
        def replace_ranges(match):
            field = match.group(1)
            start = match.group(2)
            end = match.group(3)
            return f"{field} >= {start} && {field} <= {end}"
        
        conditions = re.sub(r'(\w+) == ([\d.]+)-([\d.]+)', replace_ranges, conditions)
        
        drl.append(f"""
rule "{rule_name}"
when
    $d: Dispute( {conditions} )
then
    modify($d) {{ setAction("{action}"); }}
end""")
    
    # Add default rule for cases not covered
    drl.append(f"""
rule "Default_Rule"
when
    $d: Dispute( action == null )
then
    modify($d) {{ setAction("challenge"); }}
end""")
    
    return "\n".join(drl)

def generate_sql_case(ruleset):
    sql_lines = ["CASE"]
    
    for rule in ruleset.rules:
        condition_str, action = parse_rule_str(str(rule))
        if not condition_str or condition_str == "ELSE":
            continue
            
        # Convert RIPPER conditions to SQL format
        sql_condition = condition_str.replace("^", " AND ")
        sql_condition = sql_condition.replace("=", " = ")
        
        # Handle range conditions
        import re
        def replace_ranges(match):
            field = match.group(1)
            start = match.group(2)
            end = match.group(3)
            return f"{field} >= {start} AND {field} <= {end}"
        
        sql_condition = re.sub(r'(\w+) = ([\d.]+)-([\d.]+)', replace_ranges, sql_condition)
        
        sql_lines.append(f"  WHEN {sql_condition} THEN '{action}'")
    
    # Add default case
    sql_lines.append(f"  ELSE 'challenge'")
    sql_lines.append("END AS predicted_action")
    
    return "\n".join(sql_lines)

# -----------------------------
# 8. Output

print("\n🎯 Accuracy Results:")
print(f"XGBoost Accuracy: {xgb_acc:.4f}")
print(f"RIPPER Accuracy:  {ripper_acc:.4f}")

# Generate rule outputs
ripper_rules_output = []
for rule in ripper.ruleset_.rules:
    ripper_rules_output.append(f"- {str(rule)}")
ripper_rules_text = "\n".join(ripper_rules_output)

drools_rules_text = generate_drools(ripper.ruleset_)
sql_case_text = generate_sql_case(ripper.ruleset_)

# Save to separate files
with open('ripper_rules.txt', 'w') as f:
    f.write("RIPPER Rules Generated from XGBoost Model\n")
    f.write("=" * 50 + "\n\n")
    f.write(ripper_rules_text)
    print("\n✅ RIPPER rules saved to: ripper_rules.txt")

with open('drools_rules.drl', 'w') as f:
    f.write("// Drools Rules Generated from RIPPER\n")
    f.write("// Generated automatically from XGBoost model\n\n")
    f.write("package com.example.dispute;\n\n")
    f.write(drools_rules_text)
    print("✅ Drools rules saved to: drools_rules.drl")

with open('sql_rules.sql', 'w') as f:
    f.write("-- SQL CASE Statement Generated from RIPPER\n")
    f.write("-- Generated automatically from XGBoost model\n\n")
    f.write("SELECT *,\n")
    f.write(sql_case_text)
    f.write("\nFROM disputes;")
    print("✅ SQL rules saved to: sql_rules.sql")

print("\n📜 RIPPER Rules:")
for rule in ripper.ruleset_.rules:
    print("-", str(rule))
