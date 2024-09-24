from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the existing disease prediction model
disease_model = joblib.load('dna_disease_model.pkl')
disease_label_encoder = joblib.load('label_encoder.pkl')

# Load the new DNA reconstruction model
reconstruction_model = joblib.load('dna_reconstruction_model.pkl')

# Function to One-Hot Encode DNA Sequences
def one_hot_encode_dna(sequence, max_len):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded = [mapping[base] for base in sequence]
    
    if len(encoded) < max_len:
        encoded += [[0, 0, 0, 0]] * (max_len - len(encoded))  # Padding
    else:
        encoded = encoded[:max_len]  # Truncate to max_len
    
    return np.array(encoded).reshape(-1)

# Maximum sequence length (adjust this to match training data)
max_len = 10

# Route for the original model (disease prediction)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    dna_sequence = data.get('dna_sequence')
    encoded_sequence = one_hot_encode_dna(dna_sequence, 13)
    
    predicted_label = disease_model.predict([encoded_sequence])
    predicted_disease = disease_label_encoder.inverse_transform(predicted_label)
    
    return jsonify({'predicted_disease': predicted_disease[0]})

# Route for the new model (sequence reconstruction)
@app.route('/reconstruction', methods=['POST'])
def reconstruction():
    try:
        # Get DNA sequences from the request
        data = request.json
        seq1 = data.get('sequence1')
        seq2 = data.get('sequence2')

        # One-hot encode the sequences
        encoded_seq1 = one_hot_encode_dna(seq1, max_len)
        encoded_seq2 = one_hot_encode_dna(seq2, max_len)

        # Concatenate the two sequences
        input_data = np.hstack([encoded_seq1, encoded_seq2]).reshape(1, -1)

        # Predict the new sequence
        predicted_sequence_encoded = reconstruction_model.predict(input_data)

        # Convert the one-hot encoded sequence back to a DNA sequence
        def decode_sequence(encoded_sequence):
            base_map = ['A', 'C', 'G', 'T']
            decoded_sequence = ''.join([base_map[np.argmax(encoded_sequence[i:i+4])] for i in range(0, len(encoded_sequence), 4)])
            return decoded_sequence

        predicted_sequence = decode_sequence(predicted_sequence_encoded[0])

        return jsonify({'predicted_sequence': predicted_sequence})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
