#!/usr/bin/env python3
"""
RIPPER Model Training and Testing Script
Supports training, testing, saving, and retraining models with CSV data
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from wittgenstein import RIPPER
import warnings
warnings.filterwarnings('ignore')

class RipperModelManager:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = None

    def preprocess_data(self, df, target_col, fit_encoders=True):
        """Preprocess the data by encoding categorical variables"""
        df_processed = df.copy()

        # Separate features and target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]

        # Store column information
        if fit_encoders:
            self.feature_columns = list(X.columns)
            self.target_column = target_col

        # Encode categorical variables
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                if fit_encoders:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_vals = set(X[col].astype(str))
                        known_vals = set(self.label_encoders[col].classes_)
                        new_vals = unique_vals - known_vals

                        if new_vals:
                            # Add new categories to encoder
                            all_vals = list(known_vals) + list(new_vals)
                            self.label_encoders[col].classes_ = np.array(all_vals)

                        X[col] = self.label_encoders[col].transform(X[col].astype(str))

        # Encode target variable
        if fit_encoders:
            if 'target' not in self.label_encoders:
                self.label_encoders['target'] = LabelEncoder()
            y = self.label_encoders['target'].fit_transform(y.astype(str))
        else:
            if 'target' in self.label_encoders:
                y = self.label_encoders['target'].transform(y.astype(str))

        return X, y

    def train_model(self, X, y, **kwargs):
        """Train the RIPPER model"""
        print("Training RIPPER model...")

        # Initialize RIPPER with default parameters
        ripper_params = {
            'random_state': 42,
            'n_discretize_bins': 10,
            'max_rules': 100,
            'prune_size': 0.33,
            'dl_allowance': 64
        }
        ripper_params.update(kwargs)

        self.model = RIPPER(**ripper_params)
        self.model.fit(X, y)

        print(f"Model trained successfully with {len(self.model.ruleset_)} rules")
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")

        # Get original class names for better readability
        if 'target' in self.label_encoders:
            target_names = self.label_encoders['target'].classes_
            print(classification_report(y_test, y_pred, target_names=target_names))
        else:
            print(classification_report(y_test, y_pred))

        return accuracy

    def save_model(self, filepath):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a previously trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found!")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']

        print(f"Model loaded from {filepath}")

    def display_rules(self):
        """Display the learned rules"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        print(f"\nLearned Rules ({len(self.model.ruleset_)} rules):")
        print("=" * 50)
        for i, rule in enumerate(self.model.ruleset_, 1):
            print(f"Rule {i}: {rule}")
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='RIPPER Model Training and Testing')
    parser.add_argument('--train', type=str, help='Path to training CSV file')
    parser.add_argument('--test', type=str, help='Path to test CSV file (optional)')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--model-path', type=str, default='ripper_model.pkl',
                        help='Path to save/load model (default: ripper_model.pkl)')
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain existing model with new data')
    parser.add_argument('--predict', type=str, help='Path to CSV file for predictions')
    parser.add_argument('--show-rules', action='store_true', help='Display learned rules')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test size for train/test split (default: 0.2)')

    args = parser.parse_args()

    # Initialize model manager
    ripper_manager = RipperModelManager()

    # Handle different modes
    if args.retrain and os.path.exists(args.model_path):
        # Load existing model for retraining
        print("Loading existing model for retraining...")
        ripper_manager.load_model(args.model_path)

        if args.train:
            print(f"Retraining with new data from: {args.train}")
            df_new = pd.read_csv(args.train)
            X_new, y_new = ripper_manager.preprocess_data(df_new, args.target, fit_encoders=False)

            # Retrain the model (in RIPPER, this means training a new model)
            ripper_manager.train_model(X_new, y_new)
            ripper_manager.save_model(args.model_path)
        else:
            print("Error: --train required for retraining")
            return

    elif args.train:
        # Train new model
        print(f"Training new model with data from: {args.train}")
        df = pd.read_csv(args.train)

        # Check if target column exists
        if args.target not in df.columns:
            print(f"Error: Target column '{args.target}' not found in the dataset")
            print(f"Available columns: {list(df.columns)}")
            return

        # Preprocess data
        X, y = ripper_manager.preprocess_data(df, args.target, fit_encoders=True)

        # Split data if no separate test file provided
        if args.test:
            df_test = pd.read_csv(args.test)
            X_test, y_test = ripper_manager.preprocess_data(df_test, args.target, fit_encoders=False)
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=42, stratify=y
            )

        # Train model
        ripper_manager.train_model(X_train, y_train)

        # Evaluate model
        ripper_manager.evaluate_model(X_test, y_test)

        # Save model
        ripper_manager.save_model(args.model_path)

    elif args.predict:
        # Load model and make predictions
        if not os.path.exists(args.model_path):
            print(f"Error: Model file {args.model_path} not found!")
            return

        ripper_manager.load_model(args.model_path)

        print(f"Making predictions on: {args.predict}")
        df_pred = pd.read_csv(args.predict)

        # If target column exists, remove it for prediction
        if args.target in df_pred.columns:
            X_pred, _ = ripper_manager.preprocess_data(df_pred, args.target, fit_encoders=False)
        else:
            # Preprocess without target column
            X_pred = df_pred.copy()
            for col in X_pred.columns:
                if col in ripper_manager.label_encoders:
                    X_pred[col] = ripper_manager.label_encoders[col].transform(X_pred[col].astype(str))

        predictions = ripper_manager.predict(X_pred)

        # Convert predictions back to original labels
        if 'target' in ripper_manager.label_encoders:
            predictions = ripper_manager.label_encoders['target'].inverse_transform(predictions)

        # Save predictions
        pred_df = df_pred.copy()
        pred_df['predictions'] = predictions
        output_file = args.predict.replace('.csv', '_predictions.csv')
        pred_df.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")

    else:
        print("Error: Please provide --train for training or --predict for predictions")
        parser.print_help()
        return

    # Show rules if requested
    if args.show_rules and ripper_manager.model is not None:
        ripper_manager.display_rules()

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import wittgenstein
    except ImportError:
        print("Error: wittgenstein package not found!")
        print("Install it using: pip install wittgenstein")
        exit(1)

    main()