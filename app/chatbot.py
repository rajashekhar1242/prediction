
from flask import Flask, request, render_template, jsonify,session 
import numpy as np
import pandas as pd
import pickle
import nltk
import uuid
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



sym_des = pd.read_csv("kaggle_dataset/symptoms_df.csv")
precautions = pd.read_csv("kaggle_dataset/precautions_df.csv")
description = pd.read_csv("kaggle_dataset/description.csv")


from collections import defaultdict
from itertools import combinations

df_symptoms = pd.read_csv('kaggle_dataset/symptoms_df.csv')
related_symptoms_map = defaultdict(set)
symptom_set = set()

for _, row in df_symptoms.iterrows():
    symptoms = [s.strip().replace('_', ' ').lower() for s in row.dropna().values[1:]]  
    symptom_set.update(symptoms)
    for s1, s2 in combinations(symptoms, 2):
        related_symptoms_map[s1].add(s2)
        related_symptoms_map[s2].add(s1)


related_symptoms_map = {k: list(v) for k, v in related_symptoms_map.items()}

from tensorflow.keras.models import load_model  # type: ignore

nn_model = load_model('model/disease_prediction_enhanced.h5')
with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


symptoms_list = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

symptoms_list_processed = {symptom.replace('_', ' ').lower(): value for symptom, value in symptoms_list.items()}



stop_words = set(stopwords.words('english'))
all_symptoms_nlp = list(symptoms_list_processed.keys())
all_symptoms = list(symptom_set)



def predicted_value_nn(patient_symptoms):
    i_vector = np.zeros(len(symptoms_list_processed))  # Initialize an empty vector of zeros
    
    # Iterate over patient symptoms and update i_vector
    for i in patient_symptoms:
        symptom_key = i.strip().lower()  # Normalize symptom (remove spaces, lower case)
        if symptom_key in symptoms_list_processed:
            i_vector[symptoms_list_processed[symptom_key]] = 1
        else:
            print(f"Warning: '{symptom_key}' not found in symptoms list.")
    
    i_vector = np.array([i_vector])  # Convert to numpy array for prediction
    prediction = nn_model.predict(i_vector)
    predicted_index = np.argmax(prediction)  # Get the index of the max predicted value
    return label_encoder.inverse_transform([predicted_index])[0]  # Return predicted disease



import spacy
nlp = spacy.load("en_core_web_sm")

def extract_symptoms_from_text(text):
    doc = nlp(text.lower())
    input_text = ' '.join([token.text for token in doc if not token.is_stop and not token.is_punct])

    matched_symptoms = []
    for symptom in all_symptoms:
        if symptom in input_text:
            matched_symptoms.append(symptom)

    # If nothing matched directly, use fuzzy matching
    if not matched_symptoms:
        fuzzy_matches = correct_spelling(input_text,symptoms_list_processed)
        matched_symptoms.extend(fuzzy_matches)

    return list(set(matched_symptoms)) 

def information(predicted_dis):
    disease_desciption = description[description['Disease'] == predicted_dis]['Description']
    disease_desciption = " ".join([w for w in disease_desciption])

    disease_precautions = precautions[precautions['Disease'] == predicted_dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    disease_precautions = [col for col in disease_precautions.values]

    return disease_desciption, disease_precautions

from .utils import detect_intent,correct_spelling
greeting_keywords = ["hello", "hi", "hey"]
def process_user_message(message):
    intent = detect_intent(message)

    if 'chat_id' not in session:
        session['chat_id'] = str(uuid.uuid4())
        session['symptoms'] = []
        session['awaiting_symptom_confirmation'] = False
        session['suggested_symptoms'] = []

        if intent in greeting_keywords:
            return jsonify({
                "response": (
                "👋 Hello! I'm your virtual medical assistant. You can tell me your symptoms one at a time.<br>"
                "When you're ready, type <strong>'done'</strong> to get a diagnosis.<br><br>"
                "🧭 Want to find nearby hospitals? Just click the <strong>📍 Locate Nearby Hospitals</strong> button below anytime!"
                )
                })
        else:
            return jsonify({
                "response": (
                "👋 Hello! I'm your virtual medical assistant. You can tell me your symptoms one at a time.<br>"
                "When you're ready, type <strong>'done'</strong> to get a diagnosis.<br><br>"
                "🧭 Want to find nearby hospitals? Just click the <strong>📍 Locate Nearby Hospitals</strong> button below anytime!"
                )
                })
    # If waiting for symptom confirmation
    if session.get('awaiting_symptom_confirmation'):
        user_input = message.split(",")  # Assuming comma-separated symptoms
        confirmed = [s.strip().lower() for s in user_input if s.strip()]
        session['symptoms'].extend(confirmed)
        session['symptoms'] = list(set(session['symptoms']))
        session['awaiting_symptom_confirmation'] = False
        session['suggested_symptoms'] = []
        return jsonify({"response": f"✅ Noted: {', '.join(confirmed)}. You can add more symptoms or type 'done'."})

    if message == 'done':
        if not session['symptoms']:
            return jsonify({"response": "❗️Please share at least one symptom first."})
        
        # Now predict the disease using the symptoms collected so far
        predicted_disease = predicted_value_nn(session['symptoms'])  # Call your disease prediction function
        dis_des, precautions = information(predicted_disease)  # Get description and precautions
        
        # Prepare precautions for display
        my_precautions = [i for i in precautions[0]]

        # Construct the response with prediction, description, and precautions
        response = {"response": (
        f"🔬 Disease prediction complete: {predicted_disease}. Here's what we found:<br>\n"
        f"📝 <strong>Description:</strong> {dis_des}<br>"
        f"💊 <strong>Precautions:</strong> {', '.join(my_precautions)}<br><br>"
        f"🏥 If you'd like to find a nearby hospital, click the <strong>📍 Locate Nearby Hospitals</strong> button below!"
        )}
        

        # Reset session after prediction
        session.clear()  # Reset the session after the prediction is done

        return jsonify(response)

    else:
        # Correct spelling before extracting symptoms
        corrected_message = []
        for word in message.split():
            corrected = correct_spelling(word,symptoms_list_processed)
            if corrected:
                corrected_message.append(corrected)
            else:
                corrected_message.append(word)

        corrected_message = ' '.join(corrected_message)
        extracted = extract_symptoms_from_text(corrected_message)

        unknown_symptoms = [s for s in extracted if s not in symptoms_list_processed]
        if unknown_symptoms:
          return jsonify({"response": f"❗️The following symptoms were not recognized: {', '.join(unknown_symptoms)}. Please check the spelling or try another symptom."})

        session['symptoms'].extend(extracted)
        session['symptoms'] = list(set(session['symptoms']))

        # Get related symptoms
        suggested = set()
        for s in extracted:
            suggested.update(related_symptoms_map.get(s, []))
        suggested -= set(session['symptoms'])  # remove already known ones

        if suggested:
            session['awaiting_symptom_confirmation'] = True
            session['suggested_symptoms'] = list(suggested)
            checklist_html = ''.join([
                f'<div><label><input type="checkbox" name="symptom" value="{s}"> {s.capitalize()}</label></div>'
                for s in suggested
            ])
            checklist_html += '<button onclick="submitSelectedSymptoms()">Submit</button>'

            return jsonify({
                "response": f"🤔 Based on what you said, are you also experiencing any of these?<br>{checklist_html}"
            })

    # Default return message for other conditions (optional)
    return jsonify({"response": "❗️Sorry, something went wrong. Please try again."})

