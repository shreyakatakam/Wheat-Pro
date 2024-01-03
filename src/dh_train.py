import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

def train_model(file_path):
    # Assuming df is your DataFrame with 'Allele_Sequence' and 'PH_Pooled' columns
    df = pd.read_csv(file_path)

    # Assuming df is your DataFrame with 'Allele_Sequence' and 'PH_Pooled' columns
    sequences = df['Allele_Sequence']
    values = df['DH_Pooled']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sequences, values, test_size=0.2, random_state=100)

    # Use CountVectorizer to convert sequences into k-mers
    k = 3  # Adjust the k value as needed
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train a random forest regressor
    model = RandomForestRegressor(n_estimators=200, random_state=100)
    model.fit(X_train_vectorized, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test_vectorized)

    # Save the predictions to the specified pooled file in the models directory
    pooled_file_path = 'models/DH_Pooled_model.joblib'
    joblib.dump(predictions, pooled_file_path)

    # Print the path for confirmation
    print(f"Predictions saved to: {pooled_file_path}")

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    # Save the columns used during training
    expected_columns = pd.DataFrame(X_train_vectorized.toarray(), columns=vectorizer.get_feature_names_out()).columns
    joblib.dump(expected_columns.tolist(), 'models/DH_Pooled_columns.joblib')

    # Save the model
    model_file_path = 'models/DH_Pooled_model.joblib'
    joblib.dump(model, model_file_path)

    # Print the paths for confirmation
    print(f"Columns used during training saved to: models/DH_Pooled_columns.joblib")
    print(f"Model saved to: {model_file_path}")

# Example usage
if __name__ == "__main__":
    file_path = "allele_file.csv"
    train_model(file_path)
