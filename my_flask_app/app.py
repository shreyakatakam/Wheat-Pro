from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import json
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load your trained models
models = {}
pooled_values = ['DH_Pooled', 'GFD_Pooled', 'GNPS_Pooled', 'GWPS_Pooled', 'PH_Pooled', 'GY_Pooled']

def load_expected_columns(model_name):
    expected_columns_path = os.path.join('models/', f'{model_name}_columns.joblib')
    if not os.path.exists(expected_columns_path):
        raise FileNotFoundError(f"No such file: {expected_columns_path}")
    return joblib.load(expected_columns_path)

for pooled_value in pooled_values:
    model_path = f'models/{pooled_value}_model.joblib'
    model = joblib.load(model_path)
    models[pooled_value] = model

    # Check if the expected columns file exists before attempting to load
    expected_columns_path = os.path.join('models/', f'{pooled_value}_columns.joblib')
    if not os.path.exists(expected_columns_path):
        raise FileNotFoundError(f"No such file: {expected_columns_path}")

    expected_columns = joblib.load(expected_columns_path)

df = pd.read_csv("allele_file.csv")

if df.empty or not df.columns.any():
    raise ValueError("DataFrame is empty or does not contain any columns.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure that the expected JSON structure is received
        data = request.get_json()
        allele_sequence = data.get('alleleSequence')

        if allele_sequence is None:
            return jsonify({'error': 'Allele sequence is required.'}), 400

        # Continue with the prediction logic
        input_data_subset = pd.DataFrame({'Allele_Sequence': [allele_sequence]})
        
        # Use the same CountVectorizer used during training to encode the input
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
        vectorizer.fit(expected_columns)
        input_data_encoded = vectorizer.transform(input_data_subset['Allele_Sequence'])

        predictions = {}
        for pooled_value, model in models.items():
            prediction = model.predict(input_data_encoded)
            predictions[pooled_value] = prediction[0]

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': f"Error: {str(e)}"}), 500

@app.route('/predictions')
def show_predictions():
    allele_sequence = request.args.get('allele_sequence')
    data_str = request.args.get('data')

    try:
        data = json.loads(data_str) if data_str else None
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Error decoding JSON: {str(e)}'})

    return render_template('predictions.html', allele_sequence=allele_sequence, data=data)


# New route for About Us
@app.route('/about')
def about():
    return render_template('about.html')

# New route for Contact Us
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/input_csv')
def input_csv():
    # Replace 'your_csv_file_path.csv' with the actual path to your CSV file
    csv_file_path = 'my_flask_app/static/csv/input.csv'
    
    # Load CSV data
    df = pd.read_csv(csv_file_path)

    # Extract columns and rows
    columns = df.columns.tolist()
    rows = df.values.tolist()

    return render_template('input.html', columns=columns, rows=rows)

@app.route('/output_csv')
def output_csv():
    # Replace 'your_csv_file_path.csv' with the actual path to your CSV file
    csv_file_path = 'my_flask_app/static/csv/output.csv'
    
    # Load CSV data
    df = pd.read_csv(csv_file_path)

    # Extract columns and rows
    columns = df.columns.tolist()
    rows = df.values.tolist()

    return render_template('output.html', columns=columns, rows=rows)

if __name__ == '__main__':
    app.run(debug=True)
