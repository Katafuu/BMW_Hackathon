import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

def provide_recommendations(prediction_proba, test_df_imputed, model, model_feature_names):
    """Provide actionable recommendations based on prediction probabilities and sensor data."""
    # If the prediction is "NOK" (0), give detailed suggestions for adjustment
    if prediction_proba < 0.5:  # Example threshold for NOK
        # Get the feature importance from the trained model (for tree-based models like XGBoost or RandomForest)
        feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        
        if feature_importances is not None:
            # Sort features by their importance and pick the top ones
            important_features = sorted(zip(model_feature_names, feature_importances), key=lambda x: x[1], reverse=True)
            top_features = important_features[:3]  # Get top 3 most important features
            
            # Construct recommendations based on the important features
            recommendations = []
            for feature, importance in top_features:
                feature_value = test_df_imputed[feature].iloc[0]  # Get the value of the feature for the first row
                recommendations.append(f"Adjust sensor {feature} (current value: {feature_value}) to improve the outcome.")
            
            # Combine the recommendations
            recommendation_text = "To improve the production process, consider adjusting the following sensors:\n"
            recommendation_text += "\n".join(recommendations)
        
        else:
            recommendation_text = "Feature importance not available. General suggestion: Review production parameters."
        
        return recommendation_text
    else:
        return "Production is running smoothly, keep monitoring."


def visualize_findings(test_df_imputed, model, predictions, prediction_proba, model_feature_names, save_report=False, report_filename="classification_report.txt"):
    """Visualize model findings and performance."""
    
    # 1. Feature Importance Plot (only for tree-based models like RandomForest, XGBoost)
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        sorted_idx = feature_importances.argsort()

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(model_feature_names)), feature_importances[sorted_idx], align="center")
        plt.yticks(range(len(model_feature_names)), [model_feature_names[i] for i in sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title("Feature Importance Plot")
        plt.show()

    # 2. Prediction Probability Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(prediction_proba, kde=True, bins=30, color="blue", label="Predicted Probabilities")
    plt.axvline(x=0.5, color='red', linestyle='--', label="Threshold (0.5)")
    plt.xlabel("Predicted Probability of OK")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Probabilities")
    plt.legend()
    plt.show()

    # 3. Confusion Matrix
    cm = confusion_matrix(predictions, [1 if prob >= 0.5 else 0 for prob in prediction_proba])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["NOK", "OK"], yticklabels=["NOK", "OK"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # 4. Classification Report (Precision, Recall, F1-Score)
    class_report = classification_report(predictions, [1 if prob >= 0.5 else 0 for prob in prediction_proba], target_names=["NOK", "OK"])
    print("\nClassification Report:")
    print(class_report)

    # Optionally save the classification report to a file
    if save_report:
        with open(report_filename, "w") as f:
            f.write(class_report)
        print(f"Classification report saved to {report_filename}")

    # 5. ROC Curve (Receiver Operating Characteristic Curve)
    fpr, tpr, _ = roc_curve([1 if prob >= 0.5 else 0 for prob in prediction_proba], prediction_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    plt.show()


def test_model(filename):
    # Load the trained model
    model = joblib.load('trained_model.pkl')

    # Load the label encoder
    label_encoder = joblib.load('label_encoder.pkl')

    # Load the feature names used during training
    with open('trained_columns.txt', 'r') as f:
        model_feature_names = f.read().splitlines()

    # Load the test data
    test_df = pd.read_csv(filename)

    # Strip any leading/trailing spaces from the column names
    test_df.columns = test_df.columns.str.strip()

    # Debug: Check the columns of the test data
    print("Columns in the test dataset:")
    print(test_df.columns)

    # Ensure 'physical_part_id' is present and keep it for output
    if 'physical_part_id' not in test_df.columns:
        print("Error: 'physical_part_id' is missing from the test data")
        return

    # Save the 'physical_part_id' for later use in the output
    physical_part_ids_test = test_df['physical_part_id']

    # Remove unnecessary columns for the prediction (same as during training)
    extra_columns = [col for col in test_df.columns if col not in model_feature_names and col != 'physical_part_id']
    print("\nExtra columns in test data that will be dropped:", extra_columns)
    test_df = test_df.drop(columns=extra_columns)

    # Handle missing columns in test data (same as during training)
    missing_columns = [col for col in model_feature_names if col not in test_df.columns]
    print("\nMissing columns in test data that will be added:", missing_columns)
    for col in missing_columns:
        test_df[col] = 0  # Add missing columns with a default value (e.g., 0)

    # Ensure the columns in test data match the order of the model's expected feature names
    test_df = test_df[model_feature_names]

    # Check for missing values and impute them if necessary
    imputer = SimpleImputer(strategy="mean")
    test_df_imputed = pd.DataFrame(imputer.fit_transform(test_df), columns=test_df.columns)

    # Make predictions (probabilities)
    predictions = model.predict(test_df_imputed)
    prediction_proba = model.predict_proba(test_df_imputed)[:, 1]

    # Map predictions (0, 1) to "NOK", "OK"
    status = ['NOK' if pred == 0 else 'OK' for pred in predictions]

    # Add the status and probabilities to the DataFrame
    test_df['status'] = status
    test_df['probability_OK'] = prediction_proba

    # Apply the recommendation function based on the predicted probability and feature importance
    test_df['recommendation'] = test_df['probability_OK'].apply(
        lambda prob: provide_recommendations(prob, test_df_imputed, model, model_feature_names)
    )

    # Re-add the 'physical_part_id' for the final output
    test_df['physical_part_id'] = physical_part_ids_test

    # Prepare the output DataFrame with 'physical_part_id', 'status', 'probability_OK', and 'recommendation'
    output_df = test_df[['physical_part_id', 'status', 'probability_OK', 'recommendation']]

    # Save the output to a CSV file
    output_df.to_csv('Prediction_with_recommendations.csv', index=False)
    print("Predictions with recommendations saved to Prediction_with_recommendations.csv")

    # Call the function to visualize findings and save the classification report
    visualize_findings(test_df_imputed, model, predictions, prediction_proba, model_feature_names, save_report=True, report_filename="classification_report.txt")

test_model("cleaned_test.csv")