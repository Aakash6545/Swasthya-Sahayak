# Swasthya Sahayak

**Doctor Recommendation Service**

Automatically scrapes doctor data, predicts possible diseases from user symptoms, and recommends specialists based on ratings and availability.

---

## Features

- **Health Check Endpoint**: `/health` for service status
- **Symptom Catalog**: `/symptoms` to fetch available symptom list
- **Recommendations**: `/recommend` to get top doctor suggestions
- **Fuzzy Symptom Matching**: Corrects user typos using difflib


## Prerequisites

  - **Python 3.8+**
  - **Listed in requirements.txt**
  



## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/Aakash6545/Swasthya-Sahayak.git
   cd Swasthya-Sahayak/doctor_recommendation_service
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Start Python Service

```bash
python app.py
```
- Runs on `http://localhost:5000`

## API Endpoints

### Health Check

```
GET /health
Response:
{
  "status": "healthy"
}
```

### Get Symptoms

```
GET /symptoms
Response:
{
  "status": "success",
  "total_symptoms": 132,
  "symptoms": ["headache", "nausea", ...]
}
```

### Recommend Doctors

```
POST /recommend
Request JSON:
{
  "symptom_1": "headache",
  "symptom_2": "nausea",
  "symptom_3": "fever",
  "limit": 3       # optional
}
Response JSON:
{
  "status": "success",
  "data": {
    "predicted_disease": "Migraine",
    "recommended_specialization": "neurologist",
    "top_doctor": { ... },
    "other_doctors": [ ... ]
  }
}
```

---

## Future Scope

* **User Ratings Update:** Allow users to submit and update doctor ratings for more accurate recommendations
* **Dynamic Schedules:** Integrate real-time schedule updates from providers
* **Emergency Button:** Provide instant access to nearby healthcare facilities based on user location
* **Multilingual Chatbot:** Use NLP to let users input symptoms in their preferred language
* **Accessible UI:** Build an interface with text-to-speech and alt text for images to support users with disabilities

---

## References

* **Symptom–Disease Dataset:** Kaggle Dataset
* **Healthcare Provider Data:** Web-scraped from [DrData](https://www.drdata.in)

