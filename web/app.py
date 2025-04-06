import sqlite3
from flask import Flask, render_template, request, redirect, url_for,jsonify
import pickle
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)


DATABASE = 'users.db'

# Load the trained model and preprocessing tools
model = load_model("model/emotion_dbn.h5")

with open("model/label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# Create a database connection
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row 
    return conn

# Create the users table
def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        full_name TEXT NOT NULL,
                        email TEXT NOT NULL,
                        age INTEGER NOT NULL,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# Initialize the database
init_db()


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/registeration', methods=['GET', 'POST'])
def registeration():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        age = request.form['age']
        username = request.form['username']
        password = request.form['password']
        
        # Save user to the database
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (full_name, email, age, username, password)
            VALUES (?, ?, ?, ?, ?)
        ''', (full_name, email, age, username, password))
        conn.commit()
        conn.close()
        
        
        return redirect(url_for('login'))  

    return render_template('register.html') 


@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/login_view', methods=['GET', 'POST'])
def login_view():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Connect to SQLite database
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Check if username and password exist in the users table
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            return redirect(url_for('emotion'))  # Redirect to emotion detection page
        else:
            return """<script>
                        alert("Invalid username or password. Please try again.");
                        window.location.href = "/login";
                      </script>"""

    return render_template('login.html')

@app.route('/emotion')
def emotion():
    return render_template('emotion.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get("eeg_data", [])
        print(data)
        
        # If no EEG data provided, return an error
        if not data:
            return jsonify({"error": "No EEG data provided"}), 400
        
        # Check if data contains 'hi'
        if isinstance(data, str) and data.strip().lower() == 'hi':
            return jsonify({"emotion": 'Hi!.. \n How can I help you?'})
        
        # Convert input to NumPy array and reshape for prediction
        eeg_data = np.array(data).reshape(1, -1)
        
        # Assuming you have a scaled version or directly use the data
        eeg_data_scaled = eeg_data
        
        # Predict emotion
        predictions = model.predict(eeg_data_scaled)
        predicted_label_index = np.argmax(predictions, axis=1)[0]
        predicted_emotion = label_encoder.inverse_transform([predicted_label_index])[0]
        
        # Print the predicted emotion for debugging
        print(predicted_emotion)
        
        # Map emotions to descriptions or recommendations
        emotion_recommendations = {
            "NEUTRAL": "You are in a neutral emotional state. 🤔 Consider relaxing or engaging in activities that make you feel comfortable. 🧘‍♂️",
            "POSITIVE": "You are feeling positive! 😊 Keep up the good vibes ✨ and continue to engage in activities that enhance your well-being. 💪",
            "NEGATIVE": "It seems you're experiencing negative emotions. 😔 Try taking a break, practicing deep breathing, or talking to someone you trust. 🌿🧠"
        }

        # Get the recommendation for the predicted emotion
        recommendation = emotion_recommendations.get(predicted_emotion.upper(), "No recommendation available.")
        
        return jsonify({"emotion": predicted_emotion, "recommendation": recommendation})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
