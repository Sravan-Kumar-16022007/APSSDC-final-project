# Crime Rate Prediction (ML + Streamlit)

This project predicts **future crime rates** for Indian metropolitan cities using historical NCRB-style crime data and **machine learning regression**.

A **Streamlit** web app (`app.py`) provides an interactive interface where users can select a city and a future year to get a predicted crime rate.

---

## Features

- **Predict future crime rate** for selected cities and years
- Uses a **trained regression model (Random Forest)**
- Uses **lag features** (Prev_1, Prev_2, Prev_3) to capture temporal trends
- Interactive UI built with **Streamlit**

---

## Project Structure

- `app.py`  
  Streamlit frontend + model loading + recursive forecasting logic

- `Crime_Rate_Prediction.ipynb`  
  Notebook containing data preprocessing, exploration, model comparison, training, and serialization

- `crp.csv`  
  Dataset used for training (input features and target)

- `crime_rate_model.pkl`  
  Trained model serialized with `joblib`

- `city_encoder.pkl`  
  `LabelEncoder` serialized with `joblib` to keep city encoding consistent

---

## How It Works

### 1) Data Preprocessing

- Loads `crp.csv`
- Cleans missing/empty values
- Computes **Crime_Rate** using:
  
  `Crime_Rate = Total_Crime / Population (in Lakhs) (2011)+`

- Creates lag features per city:
  - `Prev_1` = previous year crime rate
  - `Prev_2` = crime rate from two years ago
  - `Prev_3` = crime rate from three years ago

- Encodes the `City` column using `LabelEncoder`

### 2) Model Training & Selection

The notebook compares multiple algorithms and selects **Random Forest Regressor** based on performance.

### 3) Prediction / Forecasting

The Streamlit app performs **recursive forecasting**:

1. Takes the last 3 known crime-rate values for the selected city
2. Predicts the next year crime rate
3. Shifts lag values (Prev_1, Prev_2, Prev_3) and repeats until the target year

---

## Running the Streamlit App

1. Install dependencies (example):

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn
```

2. Start the app:

```bash
streamlit run app.py
```

3. Open the shown local URL in your browser.

---

## Dataset Column Notes

The dataset is expected to contain (among others):

- `Year`
- `City`
- `Population (in Lakhs) (2011)+`
- `Total_Crime`
- (and/or lag-related columns used for training)

---

## Requirements

- Python 3.x
- Streamlit
- scikit-learn
- joblib

---

## Output

The app returns:

- A **predicted crime rate** for the chosen city and future year
- Displays the value as both:
  - a success message
  - and a Streamlit metric widget

---

## Author

SRAVAN KUMAR 
JAGADEESH
VIJAYENDHRA
SURYA NARAYANA
