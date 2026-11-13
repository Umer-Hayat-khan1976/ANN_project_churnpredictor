# Customer Churn Predictor (ANN + Streamlit)

Predicts whether a bank customer will leave, using a Keras ANN (13 > 16 > 8 > 1) and a Streamlit web app.

## Structure
- data/Churn_Modelling.csv   dataset
- train.ipynb                Steps 0-8 with "My understanding" notes, bonus work, executed outputs
- models/model.keras         trained network
- models/preprocessor.pkl    fitted ColumnTransformer (One-Hot + StandardScaler)
- app.py, app_utils.py       Streamlit app and shared inference code
- assets/                    loss curve and confusion matrix

## Run
    python -m venv venv
    venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (macOS/Linux)
    pip install -r requirements.txt
    python -m streamlit run app.py

To retrain, open train.ipynb and run all cells (it overwrites the files in models/).

## Results (test set, threshold 0.5)
Accuracy 86.15 percent, class-1 recall 0.49, class-1 precision 0.74.
