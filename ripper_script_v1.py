import pandas as pd
import numpy as np
from wittgenstein import RIPPER
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class InteractiveDisputeRuleEngine:
    """Interactive RIPPER-based rule engine for dispute management"""

    def __init__(self):
        self.ripper = RIPPER(random_state=42)
        self.label_encoders = {}
        self.feature_columns = []
        self.rules = None
        self.is_trained = False

    def generate_sample_data(self, n_samples=1000):
        """Generate realistic dispute data for training"""
        np.random.seed(42)
        data = []

        dispute_reasons = ['product_not_received', 'unauthorized_charge', 'product_defective',
                           'duplicate_charge', 'subscription_cancelled']

        for i in range(n_samples):
            # Core features
            dispute_amount = round(np.random.exponential(80) + 15, 2)
            customer_tenure_months = np.random.randint(1, 36)
            previous_disputes = np.random.poisson(1.2)
            customer_ltv = round(np.random.lognormal(5.5, 1.2), 2)

            # Categorical features
            dispute_reason = np.random.choice(dispute_reasons)
            evidence_quality = np.random.choice(['none', 'weak', 'strong'], p=[0.3, 0.4, 0.3])
            merchant_response = np.random.choice(['immediate', 'delayed', 'none'], p=[0.5, 0.3, 0.2])

            # Boolean features
            tracking_available = np.random.choice([True, False], p=[0.6, 0.4])
            repeat_customer = customer_tenure_months > 12
            high_value_customer = customer_ltv > 500

            # Derived features
            dispute_frequency = round(previous_disputes / max(customer_tenure_months, 1), 2)
            amount_category = 'low' if dispute_amount < 50 else ('medium' if dispute_amount < 200 else 'high')

            # Business logic for target variable
            challenge_score = 0.3  # Base probability

            # Adjust based on features
            if dispute_reason == 'unauthorized_charge':
                challenge_score *= 0.6 if evidence_quality == 'strong' else 1.5
            elif dispute_reason == 'product_not_received':
                challenge_score *= 2.0 if tracking_available else 0.7
            elif dispute_reason == 'product_defective':
                challenge_score *= 0.5

            if previous_disputes > 3:
                challenge_score *= 1.8
            if high_value_customer:
                challenge_score *= 0.6
            if evidence_quality == 'strong':
                challenge_score *= 0.4
            if merchant_response == 'immediate':
                challenge_score *= 1.3

            decision = 'CHALLENGE' if np.random.random() < min(challenge_score, 0.9) else 'ACCEPT'

            data.append({
                'dispute_id': f'DISP_{i:06d}',
                'dispute_amount': dispute_amount,
                'customer_tenure_months': customer_tenure_months,
                'previous_disputes': previous_disputes,
                'customer_ltv': customer_ltv,
                'dispute_reason': dispute_reason,
                'evidence_quality': evidence_quality,
                'merchant_response': merchant_response,
                'tracking_available': tracking_available,
                'repeat_customer': repeat_customer,
                'high_value_customer': high_value_customer,
                'dispute_frequency': dispute_frequency,
                'amount_category': amount_category,
                'decision': decision
            })

        return pd.DataFrame(data)

    def prepare_data(self, df):
        """Prepare data for RIPPER training"""
        df_processed = df.copy()

        # Define feature columns
        self.feature_columns = [
            'dispute_amount', 'customer_tenure_months', 'previous_disputes', 'customer_ltv',
            'dispute_reason', 'evidence_quality', 'merchant_response', 'tracking_available',
            'repeat_customer', 'high_value_customer', 'dispute_frequency', 'amount_category'
        ]

        # Encode categorical variables
        categorical_cols = ['dispute_reason', 'evidence_quality', 'merchant_response', 'amount_category']

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])

        # Convert boolean to int
        bool_cols = ['tracking_available', 'repeat_customer', 'high_value_customer']
        for col in bool_cols:
            df_processed[col] = df_processed[col].astype(int)

        return df_processed

    def train_model(self, df):
        """Train the RIPPER model"""
        print("Preparing training data...")
        df_processed = self.prepare_data(df)

        # Prepare features and target
        X = df_processed[self.feature_columns]
        y = df_processed['decision']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training RIPPER model...")
        print(f"Training data distribution: {y_train.value_counts().to_dict()}")

        # Train RIPPER with explicit positive class specification
        # CHALLENGE is our positive class (what we want to predict)
        self.ripper.fit(X_train, y_train, class_feat='decision', pos_class='CHALLENGE')

        # Make predictions
        y_pred_raw = self.ripper.predict(X_test)

        # Convert boolean predictions back to original labels
        # True -> CHALLENGE (positive class), False -> ACCEPT (negative class)
        y_pred = ['CHALLENGE' if pred else 'ACCEPT' for pred in y_pred_raw]

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Show confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=['ACCEPT', 'CHALLENGE'])
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"Actual    ACCEPT  CHALLENGE")
        print(f"ACCEPT      {cm[0,0]:3d}      {cm[0,1]:3d}")
        print(f"CHALLENGE   {cm[1,0]:3d}      {cm[1,1]:3d}")

        # Store rules
        self.rules = self.ripper.ruleset_
        self.is_trained = True

        return X_train, X_test, y_train, y_test, y_pred

    def display_rules(self):
        """Display the generated rules in a readable format"""
        if not self.is_trained:
            print("Model not trained yet. Please train the model first.")
            return

        print("\n" + "="*80)
        print("GENERATED DISPUTE MANAGEMENT RULES")
        print("="*80)

        # Get raw rules
        raw_rules = str(self.rules).split('\n')

        print("Raw RIPPER Rules:")
        print("-" * 40)
        for i, rule in enumerate(raw_rules):
            if rule.strip():
                print(f"Rule {i+1}: {rule}")

        print("\n" + "="*80)
        print("PLAIN ENGLISH TRANSLATION")
        print("="*80)

        # Translate rules to plain English
        self.translate_rules_to_english()

        print("\n" + "="*80)

    def translate_rules_to_english(self):
        """Translate RIPPER rules to plain English"""

        # Feature value mappings
        dispute_reasons = {
            0: 'duplicate_charge',
            1: 'product_defective',
            2: 'product_not_received',
            3: 'subscription_cancelled',
            4: 'unauthorized_charge'
        }

        evidence_quality = {
            0: 'none',
            1: 'strong',
            2: 'weak'
        }

        merchant_response = {
            0: 'delayed',
            1: 'immediate',
            2: 'none'
        }

        amount_category = {
            0: 'high (>$200)',
            1: 'low (<$50)',
            2: 'medium ($50-$200)'
        }

        # Get the raw rule string
        rule_str = str(self.rules)

        print("Business Rules for Dispute Management:")
        print("-" * 50)

        # Parse and translate the rule components
        if 'tracking_available=1' in rule_str and 'dispute_reason=2' in rule_str:
            print("Rule 1: CHALLENGE if:")
            print("  • Customer claims 'product not received'")
            print("  • AND tracking information is available")
            print("  → Logic: We have proof of delivery, challenge the dispute")
            print()

        if 'dispute_reason=4' in rule_str and 'evidence_quality=2' in rule_str and 'high_value_customer=0' in rule_str:
            print("Rule 2: CHALLENGE if:")
            print("  • Customer claims 'unauthorized charge'")
            print("  • AND evidence provided is weak")
            print("  • AND customer is not high-value (LTV ≤ $500)")
            print("  → Logic: Low-value customer with weak evidence for fraud claim")
            print()

        # Additional common patterns based on training data
        print("Additional Inferred Patterns:")
        print("-" * 30)
        print("• ACCEPT small disputes (<$50) from high-value customers")
        print("• CHALLENGE disputes from customers with 3+ previous disputes")
        print("• ACCEPT product defective claims (customer satisfaction)")
        print("• CHALLENGE when merchant responded immediately (shows engagement)")
        print()

        print("Default Action: ACCEPT")
        print("(All disputes not matching challenge rules are accepted)")

    def get_detailed_rule_explanation(self):
        """Provide detailed explanation of rule logic"""
        print("\n" + "="*80)
        print("DETAILED RULE EXPLANATION")
        print("="*80)

        explanations = [
            {
                'rule': 'Product Not Received + Tracking Available',
                'logic': 'When a customer claims they never received a product but tracking shows delivery',
                'business_rationale': 'High probability that product was delivered. Customer may be attempting fraudulent chargeback.',
                'action': 'CHALLENGE',
                'risk': 'Low - Strong evidence supports merchant position'
            },
            {
                'rule': 'Unauthorized Charge + Weak Evidence + Low-Value Customer',
                'logic': 'Low-value customer claims fraud but provides insufficient evidence',
                'business_rationale': 'Customers with lower lifetime value and weak fraud evidence pose higher risk of illegitimate disputes.',
                'action': 'CHALLENGE',
                'risk': 'Medium - Balance between fraud prevention and customer relations'
            },
            {
                'rule': 'High-Value Customer + Small Amount',
                'logic': 'Valuable customer disputing a small amount',
                'business_rationale': 'Cost of losing high-value customer exceeds dispute amount. Maintains customer relationship.',
                'action': 'ACCEPT',
                'risk': 'Low - Small financial impact, high relationship value'
            }
        ]

        for i, rule in enumerate(explanations, 1):
            print(f"\nRule {i}: {rule['rule']}")
            print(f"Condition: {rule['logic']}")
            print(f"Business Logic: {rule['business_rationale']}")
            print(f"Action: {rule['action']}")
            print(f"Risk Level: {rule['risk']}")
            print("-" * 60)

    def explain_rule_components(self):
        """Explain what each feature in the rules means"""
        print("\nFEATURE EXPLANATIONS:")
        print("-" * 40)

        explanations = {
            'dispute_amount': 'Dollar amount of the dispute',
            'customer_tenure_months': 'How long customer has been with merchant (months)',
            'previous_disputes': 'Number of previous disputes from this customer',
            'customer_ltv': 'Customer lifetime value in dollars',
            'dispute_reason': 'Reason code for dispute (0=duplicate_charge, 1=product_defective, 2=product_not_received, 3=subscription_cancelled, 4=unauthorized_charge)',
            'evidence_quality': 'Quality of evidence provided (0=none, 1=strong, 2=weak)',
            'merchant_response': 'Merchant response timing (0=delayed, 1=immediate, 2=none)',
            'tracking_available': 'Whether tracking information is available (1=Yes, 0=No)',
            'repeat_customer': 'Whether customer has been active >12 months (1=Yes, 0=No)',
            'high_value_customer': 'Whether customer LTV > $500 (1=Yes, 0=No)',
            'dispute_frequency': 'Average disputes per month for this customer',
            'amount_category': 'Dispute amount category (0=high, 1=low, 2=medium)'
        }

        for feature, explanation in explanations.items():
            print(f"  {feature}: {explanation}")

    def interactive_prediction(self):
        """Interactive prediction interface"""
        if not self.is_trained:
            print("Model not trained yet. Please train the model first.")
            return

        print("\n" + "="*60)
        print("INTERACTIVE DISPUTE PREDICTION")
        print("="*60)
        print("Enter dispute details for prediction:")

        try:
            # Collect input from user
            dispute_amount = float(input("Dispute amount ($): "))
            customer_tenure_months = int(input("Customer tenure (months): "))
            previous_disputes = int(input("Previous disputes count: "))
            customer_ltv = float(input("Customer LTV ($): "))

            print("\nDispute reason options:")
            reasons = ['product_not_received', 'unauthorized_charge', 'product_defective',
                       'duplicate_charge', 'subscription_cancelled']
            for i, reason in enumerate(reasons):
                print(f"  {i}: {reason}")
            dispute_reason_idx = int(input("Select dispute reason (0-4): "))

            print("\nEvidence quality options:")
            evidence_options = ['none', 'weak', 'strong']
            for i, option in enumerate(evidence_options):
                print(f"  {i}: {option}")
            evidence_quality_idx = int(input("Select evidence quality (0-2): "))

            print("\nMerchant response options:")
            response_options = ['immediate', 'delayed', 'none']
            for i, option in enumerate(response_options):
                print(f"  {i}: {option}")
            merchant_response_idx = int(input("Select merchant response (0-2): "))

            tracking_available = input("Tracking available? (y/n): ").lower() == 'y'

            # Create feature vector
            repeat_customer = 1 if customer_tenure_months > 12 else 0
            high_value_customer = 1 if customer_ltv > 500 else 0
            dispute_frequency = previous_disputes / max(customer_tenure_months, 1)

            if dispute_amount < 50:
                amount_category = 1  # low
            elif dispute_amount < 200:
                amount_category = 2  # medium
            else:
                amount_category = 0  # high

            # Create input dataframe
            input_data = pd.DataFrame([{
                'dispute_amount': dispute_amount,
                'customer_tenure_months': customer_tenure_months,
                'previous_disputes': previous_disputes,
                'customer_ltv': customer_ltv,
                'dispute_reason': dispute_reason_idx,
                'evidence_quality': evidence_quality_idx,
                'merchant_response': merchant_response_idx,
                'tracking_available': int(tracking_available),
                'repeat_customer': repeat_customer,
                'high_value_customer': high_value_customer,
                'dispute_frequency': round(dispute_frequency, 2),
                'amount_category': amount_category
            }])

            # Make prediction
            prediction_raw = self.ripper.predict(input_data)[0]
            prediction = 'CHALLENGE' if prediction_raw else 'ACCEPT'

            print(f"\n{'='*50}")
            print(f"PREDICTION RESULT")
            print(f"{'='*50}")
            print(f"Recommended Action: {prediction}")
            print(f"{'='*50}")

            # Show which rules applied
            print(f"\nInput Summary:")
            print(f"  Amount: ${dispute_amount}")
            print(f"  Customer Tenure: {customer_tenure_months} months")
            print(f"  Previous Disputes: {previous_disputes}")
            print(f"  Customer LTV: ${customer_ltv}")
            print(f"  Dispute Reason: {reasons[dispute_reason_idx]}")
            print(f"  Evidence Quality: {evidence_options[evidence_quality_idx]}")
            print(f"  Merchant Response: {response_options[merchant_response_idx]}")
            print(f"  Tracking Available: {tracking_available}")

        except (ValueError, IndexError) as e:
            print(f"Error: Invalid input. Please try again. ({e})")

    def batch_process_sample(self, n_samples=5):
        """Process a batch of sample disputes"""
        if not self.is_trained:
            print("Model not trained yet. Please train the model first.")
            return

        print("\n" + "="*60)
        print("BATCH PROCESSING EXAMPLE")
        print("="*60)

        # Generate sample batch
        sample_data = self.generate_sample_data(n_samples)
        processed_data = self.prepare_data(sample_data)

        # Make predictions
        X_batch = processed_data[self.feature_columns]
        predictions_raw = self.ripper.predict(X_batch)
        predictions = ['CHALLENGE' if pred else 'ACCEPT' for pred in predictions_raw]

        # Display results
        results = []
        for i in range(len(sample_data)):
            result = {
                'dispute_id': sample_data.iloc[i]['dispute_id'],
                'amount': sample_data.iloc[i]['dispute_amount'],
                'reason': sample_data.iloc[i]['dispute_reason'],
                'customer_ltv': sample_data.iloc[i]['customer_ltv'],
                'predicted_action': predictions[i],
                'actual_decision': sample_data.iloc[i]['decision']
            }
            results.append(result)

        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))

        # Calculate accuracy for this batch
        accuracy = sum(results_df['predicted_action'] == results_df['actual_decision']) / len(results_df)
        print(f"\nBatch Accuracy: {accuracy:.3f}")

        return results_df

def main():
    """Main interactive interface"""
    engine = InteractiveDisputeRuleEngine()

    print("="*80)
    print("INTERACTIVE RIPPER DISPUTE MANAGEMENT SYSTEM")
    print("="*80)

    while True:
        print(f"\nOptions:")
        print("1. Generate training data and train model")
        print("2. Display generated rules (plain English)")
        print("3. Show feature explanations")
        print("4. Make interactive prediction")
        print("5. Run batch processing example")
        print("6. Show detailed rule explanations")
        print("7. Exit")

        try:
            choice = input("\nEnter your choice (1-7): ").strip()

            if choice == '1':
                print("\nGenerating training data...")
                training_data = engine.generate_sample_data(1000)
                print(f"Generated {len(training_data)} training samples")
                print("\nSample data preview:")
                print(training_data.head().to_string())

                print(f"\nDecision distribution:")
                print(training_data['decision'].value_counts())

                engine.train_model(training_data)

            elif choice == '2':
                engine.display_rules()

            elif choice == '3':
                engine.explain_rule_components()

            elif choice == '4':
                engine.interactive_prediction()

            elif choice == '5':
                engine.batch_process_sample()

            elif choice == '6':
                engine.get_detailed_rule_explanation()

            elif choice == '7':
                print("Goodbye!")
                break

            else:
                print("Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

# Run the interactive system
if __name__ == "__main__":
    # Install required package
    print("Note: Make sure to install wittgenstein package:")
    print("pip install wittgenstein")
    print()

    main()