import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


class DisputeRuleEngine:
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.feature_names = None
        self.label_encoders = {}
        self.extracted_rules = []

    def generate_sample_data(self, n_samples=5000):
        """Generate realistic dispute transaction data"""
        np.random.seed(42)

        # Transaction features
        transaction_amounts = np.random.lognormal(4, 1.5, n_samples)
        transaction_amounts = np.clip(transaction_amounts, 1, 10000)

        # Customer features
        customer_age = np.random.normal(40, 15, n_samples)
        customer_age = np.clip(customer_age, 18, 80)

        account_age_days = np.random.exponential(365, n_samples)
        account_age_days = np.clip(account_age_days, 1, 3650)

        # Merchant features
        merchant_categories = np.random.choice(
            [
                "online_retail",
                "restaurant",
                "gas_station",
                "grocery",
                "entertainment",
                "travel",
                "electronics",
                "clothing",
            ],
            n_samples,
        )

        merchant_risk_scores = np.random.beta(
            2, 5, n_samples
        )  # Most merchants low risk

        # Transaction patterns
        hour_of_day = np.random.randint(0, 24, n_samples)
        day_of_week = np.random.randint(0, 7, n_samples)

        # Geographic features
        same_country = np.random.choice([True, False], n_samples, p=[0.85, 0.15])
        same_city = np.random.choice([True, False], n_samples, p=[0.70, 0.30])

        # Customer behavior
        transactions_last_30_days = np.random.poisson(8, n_samples)
        avg_transaction_amount = transaction_amounts * np.random.uniform(
            0.5, 2.0, n_samples
        )

        # Payment method
        payment_methods = np.random.choice(
            ["credit_card", "debit_card", "digital_wallet"], n_samples
        )

        # Create DataFrame
        data = pd.DataFrame(
            {
                "transaction_amount": transaction_amounts,
                "customer_age": customer_age,
                "account_age_days": account_age_days,
                "merchant_category": merchant_categories,
                "merchant_risk_score": merchant_risk_scores,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "same_country": same_country.astype(int),
                "same_city": same_city.astype(int),
                "transactions_last_30_days": transactions_last_30_days,
                "avg_transaction_amount": avg_transaction_amount,
                "payment_method": payment_methods,
            }
        )

        # Create derived features
        data["amount_vs_avg_ratio"] = (
            data["transaction_amount"] / data["avg_transaction_amount"]
        )
        data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
        data["is_night"] = (
            (data["hour_of_day"] >= 22) | (data["hour_of_day"] <= 6)
        ).astype(int)
        data["high_amount"] = (
            data["transaction_amount"] > data["transaction_amount"].quantile(0.9)
        ).astype(int)

        # Generate dispute labels based on realistic patterns
        dispute_probability = (
            0.05  # Base rate
            + 0.15 * (data["merchant_risk_score"] > 0.7)  # High-risk merchants
            + 0.10 * (data["amount_vs_avg_ratio"] > 3)  # Unusual amounts
            + 0.08 * (data["same_country"] == 0)  # Foreign transactions
            + 0.06 * (data["is_night"] == 1)  # Night transactions
            + 0.04 * (data["account_age_days"] < 30)  # New accounts
            + 0.03 * (data["transactions_last_30_days"] > 20)  # High frequency
        )

        data["is_dispute"] = np.random.binomial(
            1, np.clip(dispute_probability, 0, 0.8), n_samples
        )

        return data

    def preprocess_data(self, data):
        """Preprocess data for ML models"""
        # Encode categorical variables
        categorical_cols = ["merchant_category", "payment_method"]

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[f"{col}_encoded"] = self.label_encoders[col].fit_transform(
                    data[col]
                )
            else:
                data[f"{col}_encoded"] = self.label_encoders[col].transform(data[col])

        # Select features for modeling
        feature_cols = [
            "transaction_amount",
            "customer_age",
            "account_age_days",
            "merchant_risk_score",
            "hour_of_day",
            "day_of_week",
            "same_country",
            "same_city",
            "transactions_last_30_days",
            "amount_vs_avg_ratio",
            "is_weekend",
            "is_night",
            "high_amount",
            "merchant_category_encoded",
            "payment_method_encoded",
        ]

        X = data[feature_cols]
        y = data["is_dispute"]

        self.feature_names = feature_cols
        return X, y

    def train_ensemble_models(self, X, y):
        """Train ensemble models"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
        )
        self.rf_model.fit(X_train, y_train)

        # Gradient Boosting
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.gb_model.fit(X_train, y_train)

        # Evaluate models
        rf_pred = self.rf_model.predict(X_test)
        gb_pred = self.gb_model.predict(X_test)

        print("Random Forest Performance:")
        print(classification_report(y_test, rf_pred))
        print("\nGradient Boosting Performance:")
        print(classification_report(y_test, gb_pred))

        return X_train, X_test, y_train, y_test

    def extract_rules_from_tree(
        self, tree, feature_names, class_names=["No Dispute", "Dispute"]
    ):
        """Extract rules from a single decision tree"""
        tree_rules = export_text(tree, feature_names=feature_names)
        return tree_rules

    def extract_rules_from_forest(self, max_trees=5):
        """Extract rules from Random Forest trees"""
        rules = []

        # Get feature importances
        importances = self.rf_model.feature_importances_
        feature_importance_dict = dict(zip(self.feature_names, importances))

        print("Top Feature Importances:")
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )
        for feature, importance in sorted_features[:10]:
            print(f"{feature}: {importance:.4f}")

        # Extract rules from top performing trees
        for i in range(min(max_trees, len(self.rf_model.estimators_))):
            tree = self.rf_model.estimators_[i]
            tree_rules = self.extract_rules_from_tree(tree, self.feature_names)
            rules.append(f"Tree {i+1} Rules:\n{tree_rules}\n" + "=" * 50)

        return rules

    def create_simplified_rules(self, X, y):
        """Create simplified, interpretable rules"""
        # Train a shallow decision tree for rule extraction
        simple_tree = DecisionTreeClassifier(
            max_depth=5, min_samples_split=100, min_samples_leaf=50, random_state=42
        )
        simple_tree.fit(X, y)

        # Extract rules
        simplified_rules = self.extract_rules_from_tree(simple_tree, self.feature_names)

        return simplified_rules, simple_tree

    def generate_business_rules(self, X, y):
        """Generate human-readable business rules"""
        rules = []

        # Analyze high-risk patterns
        dispute_data = X[y == 1]
        no_dispute_data = X[y == 0]

        print("\nBusiness Rules Analysis:")
        print("=" * 50)

        # Rule 1: High amount transactions
        high_amount_dispute_rate = len(
            dispute_data[dispute_data["high_amount"] == 1]
        ) / len(dispute_data)
        high_amount_normal_rate = len(
            no_dispute_data[no_dispute_data["high_amount"] == 1]
        ) / len(no_dispute_data)

        if high_amount_dispute_rate > high_amount_normal_rate * 2:
            rule = f"Rule 1: IF transaction_amount > {X['transaction_amount'].quantile(0.9):.2f} THEN dispute_risk = HIGH"
            rules.append(rule)
            print(rule)
            print(
                f"   Dispute rate: {high_amount_dispute_rate:.2%}, Normal rate: {high_amount_normal_rate:.2%}"
            )

        # Rule 2: Foreign transactions
        foreign_dispute_rate = len(
            dispute_data[dispute_data["same_country"] == 0]
        ) / len(dispute_data)
        foreign_normal_rate = len(
            no_dispute_data[no_dispute_data["same_country"] == 0]
        ) / len(no_dispute_data)

        if foreign_dispute_rate > foreign_normal_rate * 1.5:
            rule = "Rule 2: IF same_country = FALSE THEN dispute_risk = MEDIUM"
            rules.append(rule)
            print(rule)
            print(
                f"   Dispute rate: {foreign_dispute_rate:.2%}, Normal rate: {foreign_normal_rate:.2%}"
            )

        # Rule 3: Night transactions
        night_dispute_rate = len(dispute_data[dispute_data["is_night"] == 1]) / len(
            dispute_data
        )
        night_normal_rate = len(
            no_dispute_data[no_dispute_data["is_night"] == 1]
        ) / len(no_dispute_data)

        if night_dispute_rate > night_normal_rate * 1.3:
            rule = "Rule 3: IF hour_of_day BETWEEN 22 AND 6 THEN dispute_risk = MEDIUM"
            rules.append(rule)
            print(rule)
            print(
                f"   Dispute rate: {night_dispute_rate:.2%}, Normal rate: {night_normal_rate:.2%}"
            )

        # Rule 4: High merchant risk
        high_risk_threshold = X["merchant_risk_score"].quantile(0.8)
        high_risk_dispute_rate = len(
            dispute_data[dispute_data["merchant_risk_score"] > high_risk_threshold]
        ) / len(dispute_data)
        high_risk_normal_rate = len(
            no_dispute_data[
                no_dispute_data["merchant_risk_score"] > high_risk_threshold
            ]
        ) / len(no_dispute_data)

        if high_risk_dispute_rate > high_risk_normal_rate * 1.5:
            rule = f"Rule 4: IF merchant_risk_score > {high_risk_threshold:.2f} THEN dispute_risk = HIGH"
            rules.append(rule)
            print(rule)
            print(
                f"   Dispute rate: {high_risk_dispute_rate:.2%}, Normal rate: {high_risk_normal_rate:.2%}"
            )

        # Rule 5: Unusual amount ratio
        unusual_ratio_threshold = X["amount_vs_avg_ratio"].quantile(0.9)
        unusual_dispute_rate = len(
            dispute_data[dispute_data["amount_vs_avg_ratio"] > unusual_ratio_threshold]
        ) / len(dispute_data)
        unusual_normal_rate = len(
            no_dispute_data[
                no_dispute_data["amount_vs_avg_ratio"] > unusual_ratio_threshold
            ]
        ) / len(no_dispute_data)

        if unusual_dispute_rate > unusual_normal_rate * 1.5:
            rule = f"Rule 5: IF amount_vs_avg_ratio > {unusual_ratio_threshold:.2f} THEN dispute_risk = HIGH"
            rules.append(rule)
            print(rule)
            print(
                f"   Dispute rate: {unusual_dispute_rate:.2%}, Normal rate: {unusual_normal_rate:.2%}"
            )

        return rules

    def predict_with_rules(self, transaction_data):
        """Apply rules to predict dispute risk"""
        risk_score = 0
        risk_factors = []

        # Apply ML model prediction
        ml_prediction = self.rf_model.predict_proba([transaction_data])[0][1]
        risk_score += ml_prediction * 0.6  # 60% weight to ML model

        # Apply business rules
        if transaction_data[0] > 500:  # High amount
            risk_score += 0.2
            risk_factors.append("High transaction amount")

        if transaction_data[7] == 0:  # Foreign transaction
            risk_score += 0.15
            risk_factors.append("Foreign transaction")

        if transaction_data[11] == 1:  # Night transaction
            risk_score += 0.1
            risk_factors.append("Night transaction")

        if transaction_data[3] > 0.6:  # High merchant risk
            risk_score += 0.15
            risk_factors.append("High-risk merchant")

        risk_level = "LOW"
        if risk_score > 0.7:
            risk_level = "HIGH"
        elif risk_score > 0.4:
            risk_level = "MEDIUM"

        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": risk_level,
            "ml_probability": ml_prediction,
            "risk_factors": risk_factors,
        }


# Example usage
if __name__ == "__main__":
    # Initialize the rule engine
    engine = DisputeRuleEngine()

    # Generate sample data
    print("Generating sample dispute transaction data...")
    data = engine.generate_sample_data(5000)

    print(f"Dataset shape: {data.shape}")
    print(f"Dispute rate: {data['is_dispute'].mean():.2%}")
    print("\nSample data:")
    print(data.head())

    # Preprocess data
    X, y = engine.preprocess_data(data)

    # Train ensemble models
    print("\nTraining ensemble models...")
    X_train, X_test, y_train, y_test = engine.train_ensemble_models(X, y)

    # Extract rules from Random Forest
    print("\nExtracting rules from Random Forest...")
    forest_rules = engine.extract_rules_from_forest(max_trees=2)

    # Create simplified rules
    print("\nCreating simplified decision tree rules...")
    simplified_rules, simple_tree = engine.create_simplified_rules(X, y)
    print("Simplified Rules:")
    print(simplified_rules)

    # Generate business rules
    business_rules = engine.generate_business_rules(X, y)

    # Example prediction
    print("\nExample Prediction:")
    print("=" * 50)
    sample_transaction = X_test.iloc[0].values
    prediction = engine.predict_with_rules(sample_transaction)

    print(f"Risk Score: {prediction['risk_score']:.3f}")
    print(f"Risk Level: {prediction['risk_level']}")
    print(f"ML Probability: {prediction['ml_probability']:.3f}")
    print(f"Risk Factors: {', '.join(prediction['risk_factors'])}")
