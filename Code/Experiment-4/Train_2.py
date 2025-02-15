import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os


file_1 = '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex4/battery_s_tissue_s.csv'
file_2 = '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex4/battery_n_tissue_s.csv'
file_3 = '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex4/battery_s_tissue_n.csv'
file_4 = '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex4/battery_n_tissue_n.csv'
file_5 = '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex4/tissue_s_battery_s.csv'
file_6 = '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex4/tissue_n_battery_s.csv'
file_7 = '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex4/tissue_s_battery_n.csv'
file_8 = '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex4/tissue_n_battery_n.csv'



data_1 = pd.read_csv(file_1)
data_2 = pd.read_csv(file_2)
data_3 = pd.read_csv(file_3)
data_4 = pd.read_csv(file_4)
data_5 = pd.read_csv(file_5)
data_6 = pd.read_csv(file_6)
data_7 = pd.read_csv(file_7)
data_8 = pd.read_csv(file_8)


data = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8], axis=0)

# Features for object_1: angle1, distance1
# Features for object_2: angle2, distance2
X_object_1 = data[['angle1', 'distance1']]
y_object_1 = data['object1']

X_object_2 = data[['angle2', 'distance2']]
y_object_2 = data['object2']

scaler_1 = StandardScaler()
X_object_1_scaled = scaler_1.fit_transform(X_object_1)

scaler_2 = StandardScaler()
X_object_2_scaled = scaler_2.fit_transform(X_object_2)


X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_object_1_scaled, y_object_1, test_size=0.2, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_object_2_scaled, y_object_2, test_size=0.2, random_state=42)




param_grid_1 = {
    'C': [10],
    'gamma': [1],
    'kernel': ['linear', 'rbf']
}
svc_1 = SVC()
grid_1 = GridSearchCV(svc_1, param_grid_1, refit=True, verbose=2, cv=5)
grid_1.fit(X_train_1, y_train_1)


param_grid_2 = {
    'C': [10],
    'gamma': [1],
    'kernel': ['linear', 'rbf']
}
svc_2 = SVC()
grid_2 = GridSearchCV(svc_2, param_grid_2, refit=True, verbose=2, cv=5)
grid_2.fit(X_train_2, y_train_2)




results_file = '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex4/new_approach/result/svm_results_v2.txt'
with open(results_file, 'w') as f:

    y_pred_1 = grid_1.predict(X_test_1)
    f.write("Best Parameters for Object_1:\n")
    f.write(str(grid_1.best_params_) + '\n')
    f.write("Classification Report for Object_1:\n")
    f.write(classification_report(y_test_1, y_pred_1) + '\n')
    f.write("Confusion Matrix for Object_1:\n")
    f.write(str(confusion_matrix(y_test_1, y_pred_1)) + '\n')


    y_pred_2 = grid_2.predict(X_test_2)
    f.write("Best Parameters for Object_2:\n")
    f.write(str(grid_2.best_params_) + '\n')
    f.write("Classification Report for Object_2:\n")
    f.write(classification_report(y_test_2, y_pred_2) + '\n')
    f.write("Confusion Matrix for Object_2:\n")
    f.write(str(confusion_matrix(y_test_2, y_pred_2)) + '\n')


print(f"Results saved to {results_file}")


model_path_1 = '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex4/new_approach/model/object_1/svm_model_object_1_updated.pkl'
model_path_2 = '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex4/new_approach/model/object_2/svm_model_object_2_updated.pkl'

os.makedirs(os.path.dirname(model_path_1), exist_ok=True)
joblib.dump(grid_1.best_estimator_, model_path_1)

os.makedirs(os.path.dirname(model_path_2), exist_ok=True)
joblib.dump(grid_2.best_estimator_, model_path_2)

print(f"Model for Object_1 saved to {model_path_1}")
print(f"Model for Object_2 saved to {model_path_2}")
