import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('my_blink_dataset.csv')
X = df.drop('label', axis=1)
y = df['label']

X['ear_mean'] = X.mean(axis=1)
X['ear_variance'] = X.var(axis=1)
X['ear_range'] = X.max(axis=1) - X.min(axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_svm = LinearSVC(dual=False, random_state=42)
final_svm.fit(X_scaled, y)


joblib.dump(final_svm, 'EAR_SVM.pkl')
joblib.dump(scaler, 'Scaler.pkl')

print("Final model and scaler successfully saved!")