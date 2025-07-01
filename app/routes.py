from flask import Blueprint, request, jsonify, session, render_template # type: ignore
from .chatbot import process_user_message
from .nearest_hospitals import find_nearest_hospitals
from .utils import detect_intent

main = Blueprint("main", __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@main.route('/reset')
def reset():
    session.clear()
    return jsonify({"response": "Session has been reset."})

@main.route('/predict', methods=['POST'])
def predict():
    message = request.json.get('message', '').lower()
    return process_user_message(message)

@main.route('/get_nearest_hospitals', methods=['POST'])
def get_hospitals():
    data = request.get_json()
    lat = data.get("latitude")
    lon = data.get("longitude")
    if lat is None or lon is None:
        return jsonify({'error': 'Missing coordinates'}), 400
    try:
        nearest = find_nearest_hospitals(float(lat), float(lon))
        return jsonify({'nearest_hospitals': nearest})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
