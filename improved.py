#Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data from an Excel file using openpyxl engine
data = pd.read_excel('cassavadata.xlsx', engine='openpyxl')  

# Split the data into features (X) and target (y)
X = data[['leaf', 'stem']]
y = data['yield']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
rf_predictions = rf_regressor.predict(X_test)

# Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)
dt_predictions = dt_regressor.predict(X_test)

# Support Vector Machine (SVM) Regressor
svm_regressor = SVR(kernel='linear')
svm_regressor.fit(X_train, y_train)
svm_predictions = svm_regressor.predict(X_test)

# XGBoost Regressor
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_regressor.fit(X_train, y_train)
xgb_predictions = xgb_regressor.predict(X_test)

# Evaluate the models
def evaluate_model(predictions, model_name):
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"{model_name} Mean Squared Error: {mse}")
    print(f"{model_name} R-squared: {r2}")

evaluate_model(rf_predictions, "Random Forest")
evaluate_model(dt_predictions, "Decision Tree")
evaluate_model(svm_predictions, "Support Vector Machine")
evaluate_model(xgb_predictions, "XGBoost")


# Read the content of the Python script (.py file)
#with open('improved.py', 'r') as py_file:
    #script_content = py_file.read()

# Save the trained models as .pkl files
joblib.dump(rf_regressor, 'random_forest_model.pkl')
joblib.dump(dt_regressor, 'decision_tree_model.pkl')
joblib.dump(svm_regressor, 'svm_model.pkl')
joblib.dump(xgb_regressor, 'xgboost_model.pkl')

