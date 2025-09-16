import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path="data/student-mat.csv"):
    df = pd.read_csv(path, sep=';')
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop("G3", axis=1)
    y = df["G3"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
