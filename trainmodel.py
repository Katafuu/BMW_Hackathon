import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib  # For saving the model
from scipy.stats import uniform, randint

def train_model(filename):
    # Load the training data
    train_df = pd.read_csv(filename)

    # Data Preprocessing on train.csv
    # Encode categorical columns
    label_encoder = LabelEncoder()

    # Encode 'weekday' and 'shift'
    weekday_cols = ['weekday_Monday', 'weekday_Sunday', 'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday']
    shift_cols = ['shift_Nachtschicht', 'shift_Spaetschicht']

    for col in weekday_cols + shift_cols:
        train_df[col] = label_encoder.fit_transform(train_df[col])

    # Encode 'status_OK' (target variable)
    train_df['status_OK'] = label_encoder.fit_transform(train_df['status_OK'])

    # Save the physical_part_id for later use in the output
    physical_part_ids_train = train_df['physical_part_id']

    # Drop unnecessary columns (physical_part_id, ordinal_timestamp)
    train_df.drop(columns=['physical_part_id', 'ordinal_timestamp'], inplace=True)

    # Separate features and target variable
    X = train_df.drop(columns=['status_OK'])  # Features
    y = train_df['status_OK']  # Target variable (status - OK/NOK)

    # Fill missing values for numeric columns only
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

    # Convert categorical columns to category type
    categorical_cols = ['physical_part_type_type2', 'physical_part_type_type4']
    for col in categorical_cols:
        X[col] = X[col].astype('category')

    # Split the data into training (95%) and testing (5%) sets using stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42, stratify=y
    )

    # Define the model (XGBoost)
    model = xgb.XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss', 
        enable_categorical=True  # Enable categorical handling for XGBoost
    )

    # Hyperparameter distribution for RandomizedSearchCV
    param_dist = {
        'max_depth': randint(3, 6),  
        'learning_rate': uniform(0.01, 0.1),
        'n_estimators': randint(100, 150),
        'subsample': uniform(0.8, 0.1),
        'colsample_bytree': uniform(0.8, 0.1),
        'gamma': randint(0, 1)
    }

    # Perform RandomizedSearchCV with fewer iterations and less cross-validation
    random_search = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_dist, 
        n_iter=10,  
        cv=2,  
        n_jobs=-1, 
        verbose=2, 
        random_state=42
    )
    
    random_search.fit(X_train, y_train)

    # Best parameters from RandomizedSearchCV
    print(f"Best Parameters: {random_search.best_params_}")

    # Use the best model
    best_model = random_search.best_estimator_

    # Evaluate the model on the test set
    test_accuracy = best_model.score(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save the trained model to a file
    joblib.dump(best_model, 'trained_model.pkl')  # Save the model
    joblib.dump(label_encoder, 'label_encoder.pkl')  # Save the label encoder

    # Save the columns used during training to ensure consistency during prediction
    with open('trained_columns.txt', 'w') as f:
        f.write("\n".join(X.columns))

    print("Model and label encoder saved. Columns used during training saved to 'trained_columns.txt'.")


