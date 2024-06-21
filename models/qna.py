import joblib
import numpy as np

class QnAModel:
    def __init__(self):
        self.model = joblib.load('models/vitamin_deficiency_model.pkl')
    
    def predict(self, answers):
        answers = np.array(answers).reshape(1, -1)
        prediction = self.model.predict(answers)
        confidence = max(self.model.predict_proba(answers)[0])
        return prediction[0], confidence
    
    def get_vitamin_deficiency(self, prediction):
        return prediction
