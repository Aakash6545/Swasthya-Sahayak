import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class DiseasePredictor:
    """Class to handle disease prediction based on symptoms"""
    
    def __init__(self):
        """Initialize the disease predictor"""
        self.model = None
        self.feature_columns = None
    
    def train_model(self, train_df):
        """
        Train a RandomForest classifier to predict diseases from symptoms
        
        Args:
            train_df (pandas.DataFrame): Training data with symptoms and diseases
            
        Returns:
            tuple: (trained model, feature columns)
        """
        # Drop that empty column
        drop_cols = [c for c in train_df.columns if c.startswith("Unnamed")]
        train_df = train_df.drop(columns=drop_cols)

        # Clean column names
        train_df.columns = train_df.columns.str.strip().str.lower()

        # Clean the target
        train_df['prognosis'] = train_df['prognosis'].astype(str).str.strip().str.lower()

        # Clean column names
        train_df.columns = train_df.columns.str.strip().str.lower()
        
        # Prepare features and target
        X = train_df.drop(columns=["prognosis"])
        y = train_df["prognosis"]
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model and features
        self.model = clf
        self.feature_columns = X.columns
        
        return clf, X.columns
    
    def predict_disease(self, symptom_vector):
        """
        Predict disease based on symptom vector
        
        Args:
            symptom_vector (numpy.ndarray): Vector of symptoms
            
        Returns:
            str: Predicted disease
        """
        if not self.model:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(symptom_vector)[0]
    
    def visualize_feature_importance(self, top_n=15):
        """
        Visualize feature importance
        
        Args:
            top_n (int): Number of top features to visualize
        """
        if not self.model or not self.feature_columns:
            raise ValueError("Model not trained yet")
        
        feature_imp = pd.Series(
            self.model.feature_importances_, 
            index=self.feature_columns
        ).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_imp[:top_n], y=feature_imp.index[:top_n])
        plt.title(f'Top {top_n} Most Important Symptoms')
        plt.tight_layout()
        plt.show()