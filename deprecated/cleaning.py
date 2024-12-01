import pandas as pd
import numpy as np
import os

def clean_data(file_path):
    # Load the dataset
    print("Step 0: Loading the dataset...")
    cleaned_train_data = pd.read_csv(file_path)
    print(f"Initial dataset shape: {cleaned_train_data.shape}")

    # Identify columns to modify
    columns_to_modify = []

    print("Step 4: Identifying uninformative columns to fill with 0s...")
    for column in cleaned_train_data.columns:
        # Count unique values in the column
        unique_values = cleaned_train_data[column].nunique()
        
        # If the column has only one unique value, mark it for modification
        if unique_values == 1:
            print(f"Marking column '{column}' as it has only one unique value.")
            columns_to_modify.append(column)
            continue
        
        # Check if the column is a boolean column with mostly zeros
        if cleaned_train_data[column].dtype in [np.int64, np.float64]:
            value_counts = cleaned_train_data[column].value_counts(normalize=True)
            if value_counts.get(0, 0) > 0.6:  # More than 60% zeros
                print(f"Marking column '{column}' as more than 60% of its values are 0.")
                columns_to_modify.append(column)
                continue

        # Check if the column has any missing values and mark it if all values are missing
        if cleaned_train_data[column].isnull().all():
            print(f"Marking column '{column}' as it contains only missing values.")
            columns_to_modify.append(column)

    # Fill marked columns with 0
    print("Filling marked columns with 0s...")
    cleaned_train_data[columns_to_modify] = 0

    # Move modified columns to the end of the DataFrame
    print("Rearranging columns to move modified columns to the end...")
    columns_order = [col for col in cleaned_train_data.columns if col not in columns_to_modify] + columns_to_modify
    cleaned_train_data = cleaned_train_data[columns_order]

    print("Uninformative columns filled and rearranged.")

    # Step 1: Handle missing values
    print("Step 1: Handling missing values...")
    # Numeric columns: fill missing values with the median
    numeric_columns = cleaned_train_data.select_dtypes(include=[np.number]).columns
    cleaned_train_data[numeric_columns] = cleaned_train_data[numeric_columns].fillna(cleaned_train_data[numeric_columns].median())

    # Categorical columns: fill missing values with the mode
    categorical_columns = cleaned_train_data.select_dtypes(include=['object', 'category']).columns
    cleaned_train_data[categorical_columns] = cleaned_train_data[categorical_columns].fillna(cleaned_train_data[categorical_columns].mode().iloc[0])
    print("Missing values handled.")

    # Step 2: Handle timestamp features
    print("Step 2: Handling timestamp features...")

    # Convert 'message_timestamp' to datetime
    cleaned_train_data['timestamp'] = pd.to_datetime(cleaned_train_data['message_timestamp'], errors='coerce')

    # Convert timestamp to ordinal value (milliseconds since Unix epoch)
    cleaned_train_data['ordinal_timestamp'] = cleaned_train_data['timestamp'].apply(lambda x: x.value if pd.notna(x) else np.nan)

    # Extract hour, minute, second and convert them to their own ordinal features
    cleaned_train_data['hour'] = cleaned_train_data['timestamp'].dt.hour
    cleaned_train_data['minute'] = cleaned_train_data['timestamp'].dt.minute
    cleaned_train_data['second'] = cleaned_train_data['timestamp'].dt.second

    # Create columns for each day (each unique day gets its own column)
    cleaned_train_data['day'] = cleaned_train_data['timestamp'].dt.date
    day_columns = pd.get_dummies(cleaned_train_data['day'], prefix='day')
    cleaned_train_data = pd.concat([cleaned_train_data, day_columns], axis=1)

    # Drop the original timestamp-related columns
    cleaned_train_data = cleaned_train_data.drop(['message_timestamp', 'timestamp', 'day'], axis=1)

    print("Timestamp features added.")

    # Step 3: One-hot encoding categorical columns (excluding 'physical_part_id')
    print("Step 3: One-hot encoding categorical columns...")
    # Exclude 'physical_part_id' from one-hot encoding
    categorical_columns = cleaned_train_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col != 'physical_part_id']

    # Apply one-hot encoding to the remaining categorical columns (produces 0 and 1)
    cleaned_train_data = pd.get_dummies(cleaned_train_data, columns=categorical_columns, drop_first=True)

    # After one-hot encoding, apply conversion to 0 and 1 for all new columns
    for col in cleaned_train_data.columns:
        if col not in numeric_columns and cleaned_train_data[col].dtype == bool:
            cleaned_train_data[col] = cleaned_train_data[col].astype(int)

    print("One-hot encoding completed.")
    cleaned_train_data = cleaned_train_data.drop(columns=["ordinal_timestamp"])
    # Step 5: Save the cleaned data to a CSV file (auto-generated file name)
    print("Step 5: Saving the cleaned data...")

    # Generate output file name by prefixing 'cleaned_' to the input file name
    output_file_name = 'cleaned_' + os.path.basename(file_path)
    cleaned_train_data.to_csv(output_file_name, index=False)

    print(f"Cleaned data saved to '{output_file_name}'.")

clean_data('./data/raw/test.csv')