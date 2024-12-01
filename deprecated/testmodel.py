# test_model.py

import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

def provide_recommendations(prediction_proba, test_df_imputed, model, model_feature_names):
    """Generate recommendations based on predictions."""
    if prediction_proba < 0.5:
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            important_features = sorted(zip(model_feature_names, feature_importances), key=lambda x: x[1], reverse=True)
            top_features = important_features[:3]
            recommendations = [
                f"Adjust sensor {feature} (current value: {test_df_imputed[feature].iloc[0]}) to improve the outcome."
                for feature, _ in top_features
            ]
            return "To improve the production process:\n" + "\n".join(recommendations)
        return "Feature importance not available. Review production parameters."
    return "Production is running smoothly."

def test_model(filename):
    model = joblib.load('trained_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    with open('trained_columns.txt', 'r') as f:
        model_feature_names = f.read().splitlines()

    test_df = pd.read_csv(filename)
    test_df.columns = test_df.columns.str.strip()

    if 'physical_part_id' not in test_df.columns:
        raise ValueError("'physical_part_id' column is missing in the test data")

    physical_part_ids = test_df['physical_part_id']
    test_df = test_df.drop(columns=[col for col in test_df.columns if col not in model_feature_names and col != 'physical_part_id'])

    missing_cols = [col for col in model_feature_names if col not in test_df.columns]
    for col in missing_cols:
        test_df[col] = 0

    test_df = test_df[model_feature_names]
    imputer = SimpleImputer(strategy="mean")
    test_df_imputed = pd.DataFrame(imputer.fit_transform(test_df), columns=test_df.columns)

    predictions = model.predict(test_df_imputed)
    prediction_proba = model.predict_proba(test_df_imputed)[:, 1]

    status = label_encoder.inverse_transform(predictions)
    test_df['status'] = status
    test_df['probability_OK'] = prediction_proba
    test_df['recommendation'] = test_df['probability_OK'].apply(
        lambda prob: provide_recommendations(prob, test_df_imputed, model, model_feature_names)
    )

    test_df['physical_part_id'] = physical_part_ids
    output_df = test_df[['physical_part_id', 'status', 'probability_OK', 'recommendation']]
    output_df.to_csv('Prediction_with_recommendations.csv', index=False)
    print("Predictions saved to Prediction_with_recommendations.csv.")

    return test_df_imputed, model, predictions, prediction_proba, model_feature_names

# Example usage
test_df_imputed, model, predictions, prediction_proba, model_feature_names = test_model("cleaned_test.csv")
