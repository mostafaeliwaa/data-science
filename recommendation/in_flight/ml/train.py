import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from recommendation.in_flight.features import build_inflight_training_features

if __name__ == "__main__":

    df = pd.read_csv("D:\\project grad\\marketing-service\\data\\Social_Media_Advertising_Transformed .csv")

    X, y_roi, y_conv = build_inflight_training_features(df)

    X_train, X_test, y_roi_train, y_roi_test, y_conv_train, y_conv_test = train_test_split(
        X, y_roi, y_conv,
        test_size=0.3,
        random_state=42
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ===== ROI MODEL =====
    roi_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    roi_model.fit(X_train_scaled, y_roi_train)
    roi_preds = roi_model.predict(X_test_scaled)

    roi_mae = mean_absolute_error(y_roi_test, roi_preds)
    roi_rmse = np.sqrt(mean_squared_error(y_roi_test, roi_preds))
    r2_roi = r2_score(y_roi_test, roi_preds)

    print("ROI Model:")
    print("MAE:", round(roi_mae, 3))
    print("RMSE:", round(roi_rmse, 3))
    print("ROI R2:", round(r2_roi, 3))

    # ===== CONVERSION MODEL =====
    conv_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    conv_model.fit(X_train_scaled, y_conv_train)
    conv_preds = conv_model.predict(X_test_scaled)

    conv_mae = mean_absolute_error(y_conv_test, conv_preds)
    conv_rmse = np.sqrt(mean_squared_error(y_conv_test, conv_preds))
    r2_conv = r2_score(y_conv_test, conv_preds)

    print("\nConversion Model:")
    print("MAE:", round(conv_mae, 3))
    print("RMSE:", round(conv_rmse, 3))
    print("Conversion R2:", round(r2_conv, 3))

    BASE_PATH = "recommendation/in_flight/ml/"

    joblib.dump(roi_model, BASE_PATH + "roi_model.pkl")
    joblib.dump(conv_model, BASE_PATH + "conversion_model.pkl")
    joblib.dump(scaler, BASE_PATH + "scaler.pkl")

    print("\nModels and scaler saved successfully.")
