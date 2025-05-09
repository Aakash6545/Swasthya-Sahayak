import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import random
from datetime import datetime
from difflib import get_close_matches

class DataProcessor:
    """Class to handle data processing for doctor recommendation system"""
    
    # Define disease to specialization mapping
    DISEASE_SPECIALIZATION = {
    'fungal infection':       'dermatologist',
    'allergy':                'allopathic family physician',
    'gerd':                   'gastro physician',
    'chronic cholestasis':    'gastro physician',
    'drug reaction':          'allopathic family physician',
    'peptic ulcer diseae':    'gastro physician',
    'aids':                   'physician (md medicine)',
    'diabetes':               'diabetologist',
    'gastroenteritis':        'gastro physician',
    'bronchial asthma':       'chest physician',
    'hypertension':           'cardiologist',
    'migraine':               'neurologist',
    'cervical spondylosis':   'physiotherapist',
    'paralysis (brain hemorrhage)': 'neurologist',
    'jaundice':               'gastro physician',
    'malaria':                'physician (md medicine)',
    'chicken pox':            'dermatologist',
    'dengue':                 'physician (md medicine)',
    'typhoid':                'physician (md medicine)',
    'hepatitis a':            'gastro physician',
    'hepatitis b':            'gastro physician',
    'hepatitis c':            'gastro physician',
    'hepatitis d':            'gastro physician',
    'hepatitis e':            'gastro physician',
    'alcoholic hepatitis':    'gastro physician',
    'tuberculosis':           'chest physician',
    'common cold':            'allopathic family physician',
    'pneumonia':              'chest physician',
    'dimorphic hemmorhoids(piles)': 'surgeon (ms general surgery)',
    'heart attack':           'cardiologist',
    'varicose veins':         'cardiovascular thoracic surgeon',
    'hypothyroidism':         'endocrinologist',
    'hyperthyroidism':        'endocrinologist',
    'hypoglycemia':           'endocrinologist',
    'osteoarthristis':        'rheumatologist',
    'arthritis':              'rheumatologist',
    '(vertigo) paroymsal  positional vertigo': 'ent surgeon',
    'acne':                   'dermatologist',
    'urinary tract infection':'urologist',
    'psoriasis':              'dermatologist',
    'impetigo':               'dermatologist',
    'chronic headache':       'neurologist',
    'diarrhea':               'gastro physician',
    }
    
    def __init__(self, doctors_csv_path, training_csv_path, merged_dataset_path):
        """
        Initialize the data processor
        
        Args:
            doctors_csv_path (str): Path to doctors CSV file
            training_csv_path (str): Path to training CSV file
            merged_dataset_path (str): Path to merged dataset CSV file
        """
        self.doctors_csv_path = doctors_csv_path
        self.training_csv_path = training_csv_path
        self.merged_dataset_path = merged_dataset_path
    
    def scrape_doctors_info(self, url):
        """
        Scrape doctor information from a given URL
        
        Args:
            url (str): The URL to scrape
            
        Returns:
            list: List of dictionaries containing doctor information
        """
        print(f"Accessing: {url}")
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"Failed to retrieve content. Status code: {response.status_code}")
                return []
                
            soup = BeautifulSoup(response.content, 'html.parser')
            doctor_table = soup.find('tbody')
            
            if not doctor_table:
                print(f"No doctor table found at {url}")
                return []
                
            doctor_entries = doctor_table.find_all('tr')
            doctors_info = []
            
            for entry in doctor_entries:
                doctor_info = {
                    'Name': entry.find('td', attrs={'data-title': "Name"}).text.strip(),
                    'Specialization': entry.find('td', attrs={'data-title': "Special."}).text.strip(),
                    'Degree': entry.find('td', attrs={'data-title': "Degree"}).text.strip(),
                    'State': entry.find('td', attrs={'data-title': "State"}).text.strip(),
                    'City': entry.find('td', attrs={'data-title': "City"}).text.strip()
                }
                doctors_info.append(doctor_info)
            
            return doctors_info
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []
    
    def collect_all_doctors(self):
        """
        Collect doctor information from all pages
        
        Returns:
            pandas.DataFrame: DataFrame containing all doctor information
        """
        base_url = 'https://www.drdata.in/list-doctors.php?search=Doctor&page='
        num_tabs = 67  # Number of pages to scrape
        all_doctors_info = []
        
        for page_number in range(1, num_tabs + 1):
            url = f'{base_url}{page_number}'
            doctors_info = self.scrape_doctors_info(url)
            all_doctors_info.extend(doctors_info)
        
        doctors_df = pd.DataFrame(all_doctors_info)
        
        # Save to CSV in the content folder
        try:
            doctors_df.to_csv(self.doctors_csv_path, index=False)
            print(f"Doctor information scraped and saved to {self.doctors_csv_path}")
        except Exception as e:
            print(f"Error occurred while saving to CSV: {e}")
        
        return doctors_df
    
    def load_doctors_data(self):
        """
        Load doctor data from CSV or scrape if not available
        
        Returns:
            pandas.DataFrame: Doctor information
        """
        try:
            doctors_df = pd.read_csv(self.doctors_csv_path)
            print("Loaded existing doctors data")
            return doctors_df
        except FileNotFoundError:
            print("Doctors data not found, scraping from website...")
            return self.collect_all_doctors()
    
    def create_merged_dataset(self, train_df, doctors_df):
        """
        Create a merged dataset from training data and doctor information
        
        Args:
            train_df (pandas.DataFrame): Training data with symptoms and diseases
            doctors_df (pandas.DataFrame): Doctor information
            
        Returns:
            pandas.DataFrame: Merged dataset
        """
        # Clean column names
        doctors_df.columns = doctors_df.columns.str.strip().str.lower()
        train_df.columns = train_df.columns.str.strip().str.lower()
        train_df['prognosis'] = train_df['prognosis'].str.strip().str.lower()

        
        # Print to inspect
        print("Unique diseases:", train_df['prognosis'].unique())
        print("Doctor specializations:", doctors_df['specialization'].unique())
        
        final_data = []
        
        for _, row in train_df.iterrows():
            disease = row['prognosis'].strip().lower()
            
            # Skip if disease not in mapping
            if disease not in self.DISEASE_SPECIALIZATION:
                continue
                
            specialization = self.DISEASE_SPECIALIZATION[disease]
            
            # Get symptoms (up to 3)
            symptoms = [col for col in row.index if row[col] == 1 and col != 'prognosis'][:3]
            symptoms += [''] * (3 - len(symptoms))  # Pad to always have 3
            
            # Filter matching doctors
            matching_docs = doctors_df[
                doctors_df['specialization'].str.lower() == specialization.lower()
            ]
            
            for _, doc in matching_docs.iterrows():
                final_data.append({
                    'Disease': disease.title(),
                    'Symptom_1': symptoms[0],
                    'Symptom_2': symptoms[1],
                    'Symptom_3': symptoms[2],
                    'Specialization': specialization,
                    'Name': doc['name'],
                    'State': doc['state'],
                    'City': doc['city'],
                    'user_rating': random.randint(0, 5),
                    'schedule': "10:00 AM-10:30 AM"  # Default schedule
                })
        
        final_df = pd.DataFrame(final_data)
        final_df.to_csv(self.merged_dataset_path, index=False)
        return final_df
    
    def get_merged_dataset(self):
        """
        Get merged dataset from CSV or create if not available
        
        Returns:
            pandas.DataFrame: Merged dataset
        """
        try:
            final_df = pd.read_csv(self.merged_dataset_path)
            print("Loaded existing merged dataset")
            return final_df
        except FileNotFoundError:
            print("Creating merged dataset...")
            train_df = pd.read_csv(self.training_csv_path)
            doctors_df = self.load_doctors_data()
            return self.create_merged_dataset(train_df, doctors_df)
    
    def correct_symptom_input(self, symptom, known_symptoms, cutoff=0.7):
        """
        Try to find the closest-known symptom to the user's input
        
        Args:
            symptom (str): User input symptom
            known_symptoms (list): List of known symptoms
            cutoff (float): Similarity cutoff
            
        Returns:
            str or None: Closest match or None
        """
        symptom = symptom.strip().lower()
        matches = get_close_matches(symptom, known_symptoms, n=1, cutoff=cutoff)
        return matches[0] if matches else None
    
    def symptoms_to_vector(self, symptom_list, feature_columns):
        """
        Convert symptoms to vector
        
        Args:
            symptom_list (list): List of symptoms
            feature_columns (list): List of feature column names
            
        Returns:
            numpy.ndarray: Vector of symptoms
        """
        vec = pd.Series(0, index=feature_columns)
        for raw in symptom_list:
            corrected = self.correct_symptom_input(raw, feature_columns)
            if corrected:
                vec[corrected] = 1
            else:
                print(f"Warning: '{raw}' not recognized as any known symptom.")
        return vec.values.reshape(1, -1)
    
    def recommend_doctors(self, symptom_1, symptom_2, symptom_3, model, 
                         feature_columns, doctors_df, time_slots, limit=3):
        """
        Recommend doctors based on symptoms
        
        Args:
            symptom_1 (str): First symptom
            symptom_2 (str): Second symptom
            symptom_3 (str): Third symptom
            model: Trained disease prediction model
            feature_columns (list): List of feature column names
            doctors_df (pandas.DataFrame): Doctor information
            time_slots (list): List of time slots
            limit (int): Maximum number of recommendations
            
        Returns:
            dict or str: Recommendations or error message
        """
        # Validate inputs
        if len({symptom_1, symptom_2, symptom_3}) < 3:
            return "Error: Please enter three unique symptoms."
        
        # Predict disease
        features = self.symptoms_to_vector(
            [symptom_1, symptom_2, symptom_3], 
            feature_columns
        )
        predicted_disease = model.predict(features)[0].strip().lower()
        
        # Map to specialization
        specialization = self.DISEASE_SPECIALIZATION.get(predicted_disease)
        if not specialization:
            return f"No specialist mapping found for predicted disease '{predicted_disease}'."
        
        # Find matching doctors
        doctors_df.columns = doctors_df.columns.str.strip().str.lower()
        matching = doctors_df[
            doctors_df["specialization"].str.lower() == specialization.lower()
        ]
        
        if matching.empty:
            return f"No doctors found for specialization '{specialization}'."
        
        # Build recommendations
        recommendations = []
        for _, doc in matching.iterrows():
            # Parse random schedule
            schedule_str = random.choice(time_slots)
            start, end = schedule_str.split('-')
            
            recommendations.append({
                "name": doc["name"],
                "city": doc["city"],
                "state": doc["state"],
                "rating": random.randint(3, 5),  # Biased towards higher ratings
                "schedule": schedule_str,
                "start_time": start.strip(),
                "end_time": end.strip()
            })
        
        # Sort by rating (descending)
        recommendations.sort(key=lambda x: -x["rating"])
        
        # Get top doctors
        top_doctors = recommendations[:limit]
        
        # Build response
        response = {
            "predicted_disease": predicted_disease.title(),
            "recommended_specialization": specialization,
            "top_doctor": top_doctors[0] if top_doctors else None,
            "other_doctors": top_doctors[1:] if len(top_doctors) > 1 else []
        }
        
        return response