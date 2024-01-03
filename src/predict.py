import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import os

def load_model(model_name):
    model_path = f'models/{model_name}_model.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No such file: {model_path}")
    return joblib.load(model_path)

def load_expected_columns(model_name):
    expected_columns_path = f'models/{model_name}_columns.joblib'
    if not os.path.exists(expected_columns_path):
        raise FileNotFoundError(f"No such file: {expected_columns_path}")
    return joblib.load(expected_columns_path)

def predict_for_allele_sequences(allele_sequences):
    # Load models and expected columns
    pooled_values = ['DH_Pooled', 'GFD_Pooled', 'GNPS_Pooled', 'GWPS_Pooled', 'PH_Pooled', 'GY_Pooled']
    models = {}
    predictions = {}

    for pooled_value in pooled_values:
        model = load_model(pooled_value)
        expected_columns = load_expected_columns(pooled_value)

        # Use CountVectorizer to convert sequences into k-mers
        k = 3  # Adjust the k value as needed
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
        vectorizer.fit(expected_columns)  # Fit on expected columns from training
        allele_sequences_vectorized = vectorizer.transform(allele_sequences)

        # Ensure the feature names match the expected columns
        if set(vectorizer.get_feature_names_out()) != set(expected_columns):
            raise ValueError(f"Feature names mismatch for {pooled_value}. Expected: {expected_columns}, Got: {vectorizer.get_feature_names_out()}")

        # Use the trained model to make predictions
        predictions[pooled_value] = model.predict(allele_sequences_vectorized)

    return pd.DataFrame(predictions)

# Example usage
if __name__ == "__main__":
    # Replace 'allele_file.csv' with the actual path to your allele sequence file
    allele_file_path = 'allele_file.csv'
    
    # Load allele sequences from the file
    allele_df = pd.read_csv(allele_file_path)
    
    # Extract the 'Allele_Sequence' column
    allele_sequences = allele_df['Allele_Sequence']

    # Predict values for all allele sequences
    predictions_df = predict_for_allele_sequences(allele_sequences)

    # Print or use the DataFrame containing predictions
    print(predictions_df)
    
    allele_sequence_to_predict = 'AATTGGCCCCAATTGG'
    s='AAGGTTAAGGTTCCCC'
    p=predict_for_allele_sequences([s])
    print(p)
    # Correct function name
    predictions = predict_for_allele_sequences([allele_sequence_to_predict])
    
    # Print or use the DataFrame containing predictions
    print(predictions)
