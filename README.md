# Titanic Disaster Survival Prediction

## Overview
The sinking of the Titanic on April 15, 1912, is one of the most infamous maritime disasters in history. The ship, considered "unsinkable," struck an iceberg during its maiden voyage, leading to the deaths of 1,502 of the 2,224 passengers and crew members aboard. 

This project aims to analyze passenger data and build a predictive model to determine the likelihood of survival based on various features.

## Problem Statement
Using machine learning, we will predict whether a passenger survived the Titanic disaster based on their characteristics such as age, gender, passenger class, and more.

## Dataset Description
The dataset consists of two CSV files:
- **Train.csv**: Contains data of 891 passengers, including their survival status (ground truth).
- **Test.csv**: Contains data of 418 passengers without survival labels. The goal is to predict their survival outcome.

### Features in the Dataset
| Variable | Definition | Key |
|---------------|------------|----------------|
| **PassengerId** | Unique ID assigned to each passenger | |
| **Survived** | Survival status | 0 = No, 1 = Yes |
| **Pclass** | Ticket class (proxy for socio-economic status) | 1 = 1st (Upper), 2 = 2nd (Middle), 3 = 3rd (Lower) |
| **Name** | Full name of the passenger | |
| **Sex** | Gender of the passenger | male/female |
| **Age** | Age in years (fractional if < 1; estimated values are in xx.5 format) | |
| **SibSp** | Number of siblings/spouses aboard | Sibling = brother, sister, stepbrother, stepsister; Spouse = husband, wife (mistresses and fiancés were ignored) |
| **Parch** | Number of parents/children aboard | Parent = mother, father; Child = daughter, son, stepdaughter, stepson; Some children traveled only with a nanny, so parch = 0 for them. |
| **Ticket** | Ticket number | |
| **Fare** | Passenger fare amount | |
| **Cabin** | Cabin number (if available) | |
| **Embarked** | Port of embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

## Project Workflow
1. **Exploratory Data Analysis (EDA)**
   - Understanding survival distribution
   - Visualizing relationships between survival and features (e.g., gender, class, age)
   - Handling missing values and feature engineering

2. **Data Preprocessing**
   - Filling missing values (e.g., Age, Fare, Embarked)
   - Encoding categorical variables
   - Feature scaling and transformation

3. **Model Selection & Training**
   - Algorithms used:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
     - Gradient Boosting (XGBoost, LightGBM)
   - Model evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC

4. **Hyperparameter Tuning**
   - Using GridSearchCV or RandomizedSearchCV to optimize model performance

5. **Predictions & Submission**
   - Generating predictions for the test dataset
   - Formatting and submitting predictions

6. **Model Deployment (Optional)**
   - Deploying as a Flask/FastAPI web app or using Streamlit
   - Hosting on Heroku/AWS/Azure

## How to Run the Project
### Prerequisites
Ensure you have the following dependencies installed:
```
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Steps to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Titanic-Disaster-Survival-Prediction.git
   cd Titanic-Disaster-Survival-Prediction
   ```
2. Run the Jupyter Notebook or Python script to preprocess data and train models:
   ```bash
   jupyter notebook titanic_survival_prediction.ipynb
   ```
3. Generate predictions and evaluate model performance.

