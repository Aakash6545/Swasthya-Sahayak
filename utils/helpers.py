from datetime import datetime
import random

def parse_schedule(schedule_str):
    """
    Parse a schedule string into start and end datetime objects
    
    Args:
        schedule_str (str): Schedule string in format "HH:MM AM/PM-HH:MM AM/PM"
        
    Returns:
        tuple: (start_time, end_time) as datetime objects
    """
    start, end = schedule_str.split('-')
    try:
        start_time = datetime.strptime(start.strip(), "%I:%M %p")
        end_time = datetime.strptime(end.strip(), "%I:%M %p")
        return start_time, end_time
    except ValueError as e:
        print(f"Error parsing schedule: {e}")
        # Return default values
        return datetime.strptime("10:00 AM", "%I:%M %p"), datetime.strptime("10:30 AM", "%I:%M %p")

def get_available_symptoms(train_df):
    """
    Get all available symptoms from the training data
    
    Args:
        train_df (pandas.DataFrame): Training data with symptoms and diseases
        
    Returns:
        list: List of symptoms
    """
    # Clean column names first
    train_df.columns = train_df.columns.str.strip().str.lower()
    
    # Get all columns except prognosis
    symptoms = [col for col in train_df.columns if col != 'prognosis']
    
    # Replace underscores with spaces for better readability
    symptoms = [symptom.replace('_', ' ') for symptom in symptoms]
    
    return symptoms

def generate_random_schedules(num_schedules=5):
    """
    Generate random schedule time slots
    
    Args:
        num_schedules (int): Number of schedules to generate
        
    Returns:
        list: List of schedule strings
    """
    hours = list(range(9, 18))  # 9 AM to 5 PM
    minutes = [0, 30]  # On the hour or half hour
    periods = ["AM", "PM"]
    
    schedules = []
    for _ in range(num_schedules):
        # Start time
        hour = random.choice(hours)
        minute = random.choice(minutes)
        period = "AM" if hour < 12 else "PM"
        if hour > 12:
            hour -= 12
        start = f"{hour:02d}:{minute:02d} {period}"
        
        # End time (30 minutes later)
        end_hour = hour
        end_minute = minute + 30
        end_period = period
        
        if end_minute >= 60:
            end_minute -= 60
            end_hour += 1
            if end_hour > 12:
                end_hour = 1
                if end_period == "AM":
                    end_period = "PM"
                else:
                    end_period = "AM"
        
        end = f"{end_hour:02d}:{end_minute:02d} {end_period}"
        
        schedules.append(f"{start}-{end}")
    
    return schedules