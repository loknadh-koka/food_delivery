from snowflake.snowpark.context import get_active_session
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib

session = get_active_session()
session.sql("USE DATABASE FOOD_DELIVERY_ML").collect()
session.sql("USE SCHEMA ML_SCHEMA").collect()
session.sql("USE WAREHOUSE ML_WH").collect()

pdf = session.table("DELIVERY_FEATURES").to_pandas()

TARGET = "DELIVERY_TIME"
NUMERIC_FEATURES = [
    "AGE","RATING","REST_LAT","REST_LONG","DELIVERY_LAT","DELIVERY_LONG",
    "VEHICLE_CONDITION","MULTIPLE_DELIVERIES","DISTANCE_KM"
]
CATEGORICAL_FEATURES = [
    "WEATHER_CONDITIONS","ROAD_TRAFFIC_DENSITY","TYPE_OF_ORDER",
    "TYPE_OF_VEHICLE","FESTIVAL","CITY"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X_train, X_test, y_train, y_test = train_test_split(
    pdf[FEATURES], pdf[TARGET], test_size=0.20, random_state=42
)

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), CATEGORICAL_FEATURES)
])

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
linear_model.fit(X_train, y_train)
p = linear_model.predict(X_test)
print("Linear Regression:", mean_absolute_error(y_test,p),
      np.sqrt(mean_squared_error(y_test,p)), r2_score(y_test,p))

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    ))
])
random_forest_model.fit(X_train, y_train)
p = random_forest_model.predict(X_test)
print("Random Forest:", mean_absolute_error(y_test,p),
      np.sqrt(mean_squared_error(y_test,p)), r2_score(y_test,p))

pred = X_test.copy()
pred["ACTUAL_DELIVERY_TIME"] = y_test.values
pred["PREDICTED_DELIVERY_TIME"] = p
pred["ABS_ERROR"] = abs(pred["ACTUAL_DELIVERY_TIME"] - pred["PREDICTED_DELIVERY_TIME"])
session.write_pandas(pred.reset_index(drop=True), "DELIVERY_PREDICTIONS",
                     database="FOOD_DELIVERY_ML", schema="ML_SCHEMA",
                     auto_create_table=True, overwrite=True)

joblib.dump(random_forest_model, "/tmp/delivery_time_random_forest.joblib")
