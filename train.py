from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from preprocess import load_and_preprocess

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess()

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Regressor": SVR()
}

# Save results
with open("results/metrics.txt", "w") as f:
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        result = f"{name}: RMSE={rmse:.2f}, R²={r2:.2f}\n"
        print(result.strip())
        f.write(result)
