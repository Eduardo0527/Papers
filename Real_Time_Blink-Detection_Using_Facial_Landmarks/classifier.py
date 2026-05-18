import joblib
import numpy as np
import warnings


warnings.filterwarnings("ignore", message="X does not have valid feature names")

model = joblib.load('EAR_SVM.pkl')
scaler = joblib.load('Scaler.pkl')

def predict_blink(EARs):
    ear_array = np.array(EARs)
    
    ear_mean = np.mean(ear_array)
    ear_var = np.var(ear_array, ddof=1)
    ear_range = np.max(ear_array) - np.min(ear_array)
    
    feature_vector = np.concatenate([ear_array, [ear_mean, ear_var, ear_range]]).reshape(1, -1)
    
    scaled_vector = scaler.transform(feature_vector)
    prediction = model.predict(scaled_vector)
    
    if prediction[0] == 1:
        return "Blink"
    else:
        return "Normal"