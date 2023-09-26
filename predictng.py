from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import joblib
import pandas as pd

# Load the trained model
loaded_rf_model = joblib.load('random_forest_model.pkl')
loaded_dt_model = joblib.load('decision_tree_model.pkl')
loaded_xgb_model = joblib.load('svm_model.pkl')
loaded_svm_model = joblib.load('xgboost_model.pkl')

# Prepare input data for prediction (replace 'leaf_size' and 'stem_size' with your actual feature names)
new_data = pd.DataFrame({'leaf_size': [
    159.1666667, 133.3333333, 138.6666667, 137.3333333, 138.1666667, 138.1666667, 140, 156.1666667, 155, 139.5,
    146.6666667, 139, 145, 141, 140.1666667, 145.3333333, 143.6666667, 151, 155.3333333, 141.1666667, 132.6666667,
    133.1666667, 143, 138.5, 154, 147.3333333, 150.3333333, 142, 152.8333333, 145.6666667, 145.1666667, 154.5, 147,
    142.1666667, 134, 137.1666667, 152.6666667, 151.3333333, 149.3333333, 148.8333333, 137.8333333, 142.6666667,
    146.8333333, 150.3333333, 142.5, 155.1666667, 148.6666667, 141, 148.1666667, 142.5, 149.1666667, 141.3333333,
    145.1666667, 137.6666667, 147, 142.8333333, 145, 142.3333333, 146.1666667, 149.6666667, 149.1666667, 150.1666667,
    133.5, 146.8333333, 138.6666667, 145.5, 149.5, 145, 147.5, 144.8333333, 140.1666667, 145.3333333, 138.6666667,
    142.5, 142.8333333, 141.3333333, 152.8333333, 148.1666667, 153.8333333, 142.5, 136.1666667, 148, 154.5, 137.8333333,
    135.8333333, 147, 150, 149.5, 150.8333333, 142.5, 143.6666667, 148, 148.6666667, 143, 145, 142.6666667, 139.1666667,
    127, 142.3333333, 143.1666667
],
    'stem_size': [
        72, 104, 81, 79, 73, 93, 89, 68, 65, 67, 67, 98, 70, 92, 99, 73, 84, 75, 65, 98, 78, 88, 96, 76, 91, 89, 78,
        85, 82, 71, 73, 83, 73, 95, 67, 97, 99, 85, 82, 91, 77, 100, 92, 86, 93, 87, 72, 94, 67, 70, 93, 85, 84, 66, 75,
        83, 86, 73, 63, 85, 80, 83, 84, 76, 66, 93, 95, 86, 74, 65, 59, 75, 84, 69, 69, 87, 80, 94, 94, 79, 88, 80, 75, 84,
        85, 65, 58, 86, 92, 87, 67, 96, 72, 86, 94, 75, 79, 63, 80, 93
    ]
})

# Make predictions using each loaded model
rf_yield_prediction = loaded_rf_model.predict(new_data)
dt_yield_prediction = loaded_dt_model.predict(new_data)
svm_yield_prediction = loaded_svm_model.predict(new_data)
xgb_yield_prediction = loaded_xgb_model.predict(new_data)

# Display the predicted yield for each model
print("Random Forest Predicted Yield:", rf_yield_prediction)
print("Decision Tree Predicted Yield:", dt_yield_prediction)
print("Support Vector Machine Predicted Yield:", svm_yield_prediction)
print("XGBoost Predicted Yield:", xgb_yield_prediction)
