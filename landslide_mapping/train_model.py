# train_model.py
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_pipeline():
    print("🛰️ Loading your cleaned geospatial matrix (.pkl)...")

    # 1. Read the pickle data you just generated
    with open("cleaned_geospatial_matrix.pkl", "rb") as file:
        data = pickle.load(file)

    # 2. Separate Features and Target Labels
    X = data[["Slope_Angle", "Soil_Moisture", "InSAR_Displacement"]]
    y = data["Landslide_Susceptibility"]

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train the Ensemble Random Forest Model
    print("🤖 Training Spatial Classifier Engine...")
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate
    predictions = model.predict(X_test)
    print(
        f"✅ Realized Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%"
    )
    print("\nClassification Report:\n", classification_report(y_test, predictions))

    # 6. Save the trained model binary and a sample csv for the frontend map
    print("💾 Serializing model to 'landslide_model.pkl'...")
    with open("landslide_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Export a CSV copy of the whole dataframe for Streamlit to read easily
    data.to_csv("uttarakhand_geospatial_data.csv", index=False)
    print("🚀 All assets locked and saved!")


if __name__ == "__main__":
    train_pipeline()