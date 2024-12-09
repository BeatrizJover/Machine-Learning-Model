# Immo Eliza Project

## Belgium Real Estate Price Prediction Model

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [Timeline](#timeline)

## Description

This is the third stage of a larger project for predicting sell prices of real estate properties in Belgium. The previous stages performed data cleaning and exploratory data analysis and it was used for the data engineering in this stage. The data used for the project comes from [ImmoWeb](https://www.immoweb.be/en).

This project is a machine learning pipeline designed to predict real estate property prices based on various features such as room space, building state, location, and more. The pipeline processes raw data, trains a regression model, evaluates its performance, and provides visual insights into the feature importance using SHAP.

The goal is to build an accurate predictive model for real estate prices using machine learning. It focuses on leveraging structured data from real estate properties (such as area, type, and location) and using the XGBoost model for regression tasks. The project also incorporates SHAP (Shapley Additive Explanations) to provide interpretability to the model by identifying the most important features that influence predictions.

The first step in the pipeline is data preprocessing. This includes:

- Loading the dataset: Raw real estate data is loaded from `belgium_properties_data.csv`.
- Data Cleaning: Missing values, duplicates, and erroneous entries are handled.
- Feature Engineering: New features are created, such as proximity to major cities and geographical coordinates; using `codes-ins-nis-postaux-belgique.csv`
- Outlier Removal: Outliers in the dataset are identified and removed to improve model performance.
- Feature Transformation: Features are scaled using RobustScaler to ensure that no feature dominates the others due to differences in scale.
- Saving the preprocess dataset: `preprocess_properties_data.csv`.

The core of the project is the training of an XGBoost regression model. Key steps include:

- Hyperparameter Tuning: `RandomizedSearchCV` is used to find the best hyperparameters for the model.
- Model Evaluation: After training, the model's performance is evaluated using metrics like RMSE, R², MAE, MAPE and sMAPE on both the training and test datasets.
- SHAP Analysis: SHAP analysis is performed to interpret the model's predictions and identify which features have the most impact on the target variable.

After training the model, evaluation metrics are printed. Additionally, the following visualizations are generated:

- Predicted vs Actual Plot: A plot comparing actual and predicted property prices.
- SHAP Summary Plot: A visualization that shows how each feature contributes to the predictions.

### Folder and files structure

`data/` : Contains the raw and preprocessed real estate property datasets. Also a `csv` from [Open Data](https://opendata.bruxelles.be/explore/dataset/codes-ins-nis-postaux-belgique/information) used for latitude and longitude of Postal Codes.

`preprocessing.py`: Script for data cleaning, merging, feature extraction, and transformation.

`modeling.py` : Script for training the regression model and evaluating its performance.

`results/` : Contains the visualizations and evaluation outputs from the model.

`main.py` : The main entry point to run the end-to-end pipeline, from data processing to model training and evaluation.

`detailed_info/`: Contains `Project_Description.md` with detailed info of the project.

## Installation

1. Install dependencies:

- ```Python 3.12.4```
- ```pip install pandas numpy sckikit-learn matplotlib xgboost shap scipy```

## Usage

- Execute the script by running the command `python main.py` in the terminal. Load the raw dataset from the provided CSV file.Preprocess the data (remove outliers, scale features, etc.). Train the XGBoost regression model. Evaluate the model’s performance and print performance metrics. Generate visualizations including SHAP plots for feature importance.

Upon running the project, the following outputs will be produced:

Performance Metrics: RMSE, R², MAE, MAPE, etc.
Visualization Plots: Predicted vs actual prices plot, SHAP summary, bar plot, and dependence plot.

### Future Improvements

- Model Improvement: Other regression models (e.g., LightGBM, Random Forest) could be explored, and hyperparameters could be fine-tuned further for better results.

- Cross-validation: Implementing k-fold cross-validation would improve the robustness of the evaluation.

- Ensemble Methods: Combining different models could create an ensemble approach that may perform better on unseen data.

## Contributors

- [BeatrizJover](https://github.com/BeatrizJover)

## Timeline

- This stage of the project lasted 6 days in the week of 09/12/2024 16:30.
