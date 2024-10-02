import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_filename = 'random_forest_model.joblib'
rf_model = joblib.load(model_filename)

# Define the symptoms list (assuming the same order as in your dataset)
symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
            'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
            'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
            'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
            'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
            'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
            'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
            'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
            'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
            'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
            'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
            'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
            'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
            'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
            'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
            'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
            'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
            'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
            'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite',
            'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
            'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
            'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1',
            'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
            'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
            'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

# Initialize session state for checkbox states if not already initialized
if 'checkbox_states' not in st.session_state:
    st.session_state.checkbox_states = {symptom: False for symptom in symptoms}

# Streamlit UI
st.title("Disease Prediction")

# Symptoms Input
st.header("Symptoms Input")
search_query = st.text_input("Search symptoms", "", on_change=None)

# Filter symptoms based on search query
filtered_symptoms = [symptom for symptom in symptoms if search_query.lower() in symptom.lower()]

# Display checkboxes for symptoms in a grid
cols = st.columns(4)
for i, symptom in enumerate(filtered_symptoms):
    col = cols[i % 4]
    if col.checkbox(symptom, key=symptom, value=st.session_state.checkbox_states[symptom]):
        st.session_state.checkbox_states[symptom] = True
    else:
        st.session_state.checkbox_states[symptom] = False

# Function to predict disease
def predict_disease(input_df):
    prediction = rf_model.predict(input_df)
    return prediction[0]

# Make prediction on button click
if st.button("Predict"):
    user_input_dict = {symptom: value for symptom, value in st.session_state.checkbox_states.items()}
    user_input_df = pd.DataFrame([user_input_dict])
    
    # Prediction
    result = predict_disease(user_input_df)
    st.success(f"The predicted disease is: {result}")

    # Display a generic message instead of AI-generated response
    st.info("Please consult with a healthcare professional for more information about this condition.")
