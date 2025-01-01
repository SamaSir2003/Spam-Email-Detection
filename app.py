from flask import Flask, request, jsonify, render_template
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the model and feature extractor
model = pickle.load(open("model.pkl", "rb"))
feature_extractor = pickle.load(open("f_ext.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')  # Serve a simple form for user input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        email_text = request.form['email']
        # Extract features using the feature extractor
        features = feature_extractor.transform([email_text])
        prediction = model.predict(features)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
