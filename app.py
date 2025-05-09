from flask import Flask, request, jsonify
import os
import pandas as pd
import traceback

# Import custom modules
from models.disease_predictor import DiseasePredictor
from models.data_processor import DataProcessor
from utils.helpers import get_available_symptoms, parse_schedule

# Initialize Flask application
app = Flask(__name__)

# Constants and configurations
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DOCTORS_CSV_PATH = os.path.join(DATA_DIR, 'doctors_info.csv')
TRAINING_CSV_PATH = os.path.join(DATA_DIR, 'Training.csv')
MERGED_DATASET_PATH = os.path.join(DATA_DIR, 'final_merged_dataset.csv')

# Time slots for doctor appointments
TIME_SLOTS = [
    "10:00 AM-10:30 AM", "12:30 PM-01:00 PM", "01:30 PM-02:00 PM",
    "03:30 PM-04:00 PM", "07:00 PM-07:30 PM", "02:30 PM-03:00 PM"
]

# Global variables
disease_model = None
doctors_df = None
feature_columns = None
data_processor = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Return available symptoms list"""
    try:
        symptoms = get_available_symptoms(pd.read_csv(TRAINING_CSV_PATH))
        return jsonify({
            "status": "success",
            "total_symptoms": len(symptoms),
            "symptoms": symptoms
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/recommend', methods=['POST'])
def recommend_doctors():
    """
    Get doctor recommendations based on symptoms
    
    Expected JSON payload:
    {
        "symptom_1": "string",
        "symptom_2": "string",
        "symptom_3": "string",
        "limit": int (optional, default: 3)
    }
    """
    global disease_model, doctors_df, feature_columns
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        # Extract symptoms
        symptom_1 = data.get('symptom_1')
        symptom_2 = data.get('symptom_2')
        symptom_3 = data.get('symptom_3')
        limit = data.get('limit', 3)  # Default limit to 3 doctors
        
        # Validate inputs
        if not all([symptom_1, symptom_2, symptom_3]):
            return jsonify({
                "status": "error", 
                "message": "Please provide all three symptoms"
            }), 400
        
        if len({symptom_1, symptom_2, symptom_3}) < 3:
            return jsonify({
                "status": "error",
                "message": "Please provide three unique symptoms"
            }), 400
        
        # Make prediction
        recommendations = data_processor.recommend_doctors(
            symptom_1, symptom_2, symptom_3, 
            disease_model, feature_columns, 
            doctors_df, TIME_SLOTS, limit
        )
        
        if isinstance(recommendations, str):
            return jsonify({
                "status": "error",
                "message": recommendations
            }), 400
        
        return jsonify({
            "status": "success",
            "data": recommendations
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.before_request
def initialize_service():
    """Initialize the service by loading models and data"""
    global disease_model, doctors_df, feature_columns, data_processor
    
    print("Initializing doctor recommendation service...")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize data processor
    data_processor = DataProcessor(
        DOCTORS_CSV_PATH, 
        TRAINING_CSV_PATH, 
        MERGED_DATASET_PATH
    )
    
    # Load or collect doctor information
    doctors_df = data_processor.load_doctors_data()
    
    # Load or create merged dataset
    data_processor.get_merged_dataset()
    
    # Train disease prediction model
    disease_predictor = DiseasePredictor()
    disease_model, feature_columns = disease_predictor.train_model(
        pd.read_csv(TRAINING_CSV_PATH)
    )
    
    print("Service initialization complete!")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)