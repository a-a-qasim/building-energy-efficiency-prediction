
# Building Energy Efficiency Prediction Using Machine Learning

## Project Overview

A machine learning project developed to predict building heating and cooling loads using architectural characteristics.

The project applies data preprocessing, exploratory data analysis, regression modelling, hyperparameter optimisation, and model evaluation. An interactive Streamlit application provides predictions based on user-defined building characteristics.

## Dataset

**Source:** UCI Machine Learning Repository — Energy Efficiency Dataset

The dataset contains 768 simulated building configurations, eight input features, and two target variables.

### Input Features

1. Relative Compactness
2. Surface Area
3. Wall Area
4. Roof Area
5. Overall Height
6. Orientation
7. Glazing Area
8. Glazing Area Distribution

### Prediction Targets

- Heating Load
- Cooling Load

## Methodology

The machine learning workflow includes:

1. Data exploration and visualisation
2. Data cleaning and preprocessing
3. Training and testing data separation
4. Regression model comparison using cross-validation
5. Gradient Boosting hyperparameter tuning using GridSearchCV
6. Evaluation on unseen test data
7. Permutation feature importance analysis
8. Model saving and deployment with Streamlit

## Model Performance

The final Gradient Boosting models achieved the following results on the held-out test dataset:

| Metric | Heating Load | Cooling Load |
|--------|-------------:|-------------:|
| MAE | 0.3464 | 1.0221 |
| RMSE | 0.4807 | 1.4552 |
| R² | 0.9978 | 0.9771 |

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook

## Streamlit Application

The interactive dashboard includes:

- Building characteristics input fields
- Input validation
- Heating and cooling load predictions
- Prediction comparison chart
- Model performance metrics
- Prediction summary
- CSV result export
- Model information and limitations

## Project Structure

    building-energy-efficiency-prediction/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    ├── data/
    ├── notebooks/
    ├── models/
    │   ├── heating_model.pkl
    │   └── cooling_model.pkl

## Installation and Usage

Clone or download the repository and navigate to the project directory.

Create and activate a virtual environment:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install the required libraries:

    python -m pip install -r requirements.txt

Launch the Streamlit application:

    python -m streamlit run app.py

Open the local URL displayed in the terminal.

## Limitations

The models were trained on simulated building configurations from the UCI Energy Efficiency dataset.

Their performance on real buildings has not been validated. Predictions should not be interpreted as actual electricity consumption, energy bills, or verified building efficiency ratings.

## Future Improvements

- Validation using real-world building datasets
- Improved handling of physically consistent building configurations
- Extended building and environmental features
- Enhanced prediction visualisations
