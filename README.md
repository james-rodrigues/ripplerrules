## Key Features:

* Training: Train new RIPPER models from CSV data
* Testing: Evaluate model performance with separate test data or automatic train/test split
* Model Persistence: Save and load trained models with all preprocessors
* Retraining: Retrain existing models with new data
* Prediction: Make predictions on new CSV files
* Rule Display: Show the learned rules in human-readable format

## Installation Requirements:
First, install the required packages:
```
pip install pandas scikit-learn wittgenstein numpy
```
Usage Examples:
1. Train a new model:
   ```python ripper_script.py --train data.csv --target class_column```
2. Train with separate test file:
   ```python ripper_script.py --train train.csv --test test.csv --target class_column```
3. Retrain existing model:
   ```python ripper_script.py --retrain --train new_data.csv --target class_column```
4. Make predictions:
   ```python ripper_script.py --predict new_data.csv --target class_column```
5. Show learned rules:
   ```python ripper_script.py --train data.csv --target class_column --show-rules```
   
## Script Features:

* Automatic preprocessing: Handles categorical variables with label encoding
* Robust error handling: Manages missing files and column issues
* Flexible target column: Specify any column as the target
* Model persistence: Saves models with all necessary preprocessing information
* Performance metrics: Provides accuracy, classification report, and confusion matrix
* Rule visualization: Displays the learned rules in readable format

