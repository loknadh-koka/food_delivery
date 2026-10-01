# Food Delivery Time Prediction

End-to-end ML project using a Zomato food-delivery dataset, Snowflake, Snowpark Python, scikit-learn, and Streamlit.

**Problem:** Regression  
**Target:** Delivery time in minutes  
**Models:** Linear Regression, Random Forest Regressor  
**Deployed model:** Random Forest Regressor

## Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 4.955 | 6.242 | 0.558 |
| Random Forest | **3.193** | **4.065** | **0.813** |

The Random Forest was deployed based on the held-out test-set metrics.

## Workflow

1. Load the raw CSV into Snowflake.
2. Clean and cast raw fields.
3. Engineer delivery distance.
4. Split data 80/20.
5. Train Linear Regression and Random Forest models.
6. Evaluate with MAE, RMSE, and R².
7. Save the Random Forest model as a Joblib artifact.
8. Store the artifact in a Snowflake stage.
9. Deploy a Snowflake Streamlit prediction app.

## Snowflake objects

- Database: `FOOD_DELIVERY_ML`
- Schema: `ML_SCHEMA`
- Warehouse: `ML_WH`
- Raw table: `DELIVERY_DATA`
- Clean table: `DELIVERY_ML_DATA`
- Feature table: `DELIVERY_FEATURES`
- Prediction table: `DELIVERY_PREDICTIONS`
- Model stage: `MODEL_STAGE`
- Streamlit app: `FOOD_DELIVERY_PREDICTOR`

## Repository structure

```text
food-delivery-ml/
├── README.md
├── sql/
│   └── snowflake_setup.sql
├── src/
│   ├── model_training.py
│   └── streamlit_app.py
├── requirements.txt
└── .gitignore
```

The raw dataset and trained model binary are intentionally not committed to GitHub. Add the original Kaggle dataset URL and your deployed Snowflake Streamlit URL before submission.

## Author

Add your name and LinkedIn/GitHub profile here.
