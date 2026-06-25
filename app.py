import streamlit as st  # Import Streamlit to build the web UI
import pandas as pd  # Import pandas to handle tabular data (CSV)
import joblib  # Import joblib to load the trained ML model and encoder

# Load the trained Random Forest model from disk (this is the "client" app's prediction engine)
model = joblib.load("crime_rate_model.pkl")  # Loads the serialized ML model object

# Load the fitted LabelEncoder so the app can convert city names to the numeric city codes the model expects
encoder = joblib.load("city_encoder.pkl")  # Loads the serialized encoder object

# Load the historical dataset used by the app to get the latest values per city
df = pd.read_csv("crp.csv")  # Reads the CSV file into a pandas DataFrame

# Remove columns that are completely empty (all values are NaN)
df = df.dropna(axis=1, how="all")  # Drops entirely-empty columns to clean the dataset

# Remove rows that are completely empty (all values are NaN)
df = df.dropna(how="all")  # Drops entirely-empty rows to clean the dataset

# Convert the City column from city name strings to numeric city codes using the loaded encoder
df["City"] = encoder.transform(df["City"])  # Encodes city names so they match training

# Configure Streamlit page settings (title shown in browser/tab)
st.set_page_config(
    page_title="Crime Rate Predictor",  # Sets the page title
)  # Ends set_page_config configuration

# Display the main app title on the UI
st.title("🚔 Crime Rate Prediction System")  # Renders a big header in the Streamlit app

# Display a subtitle explaining what cities are included
st.subheader("Indian Metropolitan Cities")  # Renders a smaller description

# Provide a dropdown to let the user choose a city for which to forecast crime rate
city = st.selectbox(
    "Select City",  # Label for the dropdown
    encoder.classes_,  # Supplies the city names that the encoder knows
)  # Stores the selected city name into the variable "city"

# Provide a number input to let the user choose the future year to predict
target_year = st.number_input(
    "Enter Future Year",  # Prompt shown to the user
    min_value=2025,  # Smallest allowed forecast year
    max_value=2050,  # Largest allowed forecast year
    value=2025,  # Default forecast year on first load
)  # Stores the user's chosen year into "target_year"

# Define a function that forecasts the crime rate for a chosen city up to a target year
def forecast_crime(city_name, future_year):  # Takes city name and target year as inputs

    # Convert city name to the numeric code required by the model
    city_code = encoder.transform([city_name])[0]  # Encodes one city name and extracts the first code

    # Filter the full dataset to only rows for the chosen city code
    city_data = df[df["City"] == city_code]  # Keeps only the selected city's historical rows

    # Sort the city's historical data by year so "latest" values are accurate
    city_data = city_data.sort_values("Year")  # Orders by increasing year

    # Find the latest year present in the dataset for that city
    latest_year = int(city_data["Year"].max())  # Gets max year and converts to int

    # Get the population for the latest year (the model uses population as a feature)
    population = city_data.iloc[-1]["Population (in Lakhs) (2011)+"]  # Reads population from the last row

    # Extract the list of historical crime rates for the city
    crime_rates = city_data["Crime_Rate"].tolist()  # Converts the Crime_Rate column to a Python list

    # If there isn't enough history, the 3-lag recursive prediction cannot run
    if len(crime_rates) < 3:  # Ensures we have Prev_1, Prev_2, Prev_3 values
        return None  # Signals the caller that prediction is not possible

    # Set the most recent three crime rates to use as the starting lag features
    prev1 = crime_rates[-1]  # Crime rate at the latest year (Prev_1)
    prev2 = crime_rates[-2]  # Crime rate one year earlier (Prev_2)
    prev3 = crime_rates[-3]  # Crime rate two years earlier (Prev_3)

    # Start forecasting from the latest historical year
    current_year = latest_year  # Initialize the loop year to the latest available year

    # Initialize prediction variable (will be updated each loop step)
    prediction = prev1  # Default prediction starts as the last known crime rate

    # Keep predicting year-by-year until reaching the requested future year
    while current_year < future_year:  # Continue until current_year reaches future_year

        # Build a one-row DataFrame containing the model features for the next year
        input_df = pd.DataFrame({  # Creates the input feature table for the model

            "Year": [current_year + 1],  # Sets the feature "Year" to next year
            "City": [city_code],  # Sets the feature "City" to numeric city code
            "Population (in Lakhs) (2011)+": [population],  # Uses the last known population as constant feature
            "Prev_1": [prev1],  # Sets lag-1 feature to most recent crime rate
            "Prev_2": [prev2],  # Sets lag-2 feature to crime rate from one year before that
            "Prev_3": [prev3],  # Sets lag-3 feature to crime rate from two years before that

        })  # Ends DataFrame construction

        # Predict the crime rate for the next year using the trained ML model
        prediction = model.predict(input_df)[0]  # Produces one number prediction and extracts it

        # Shift lag variables forward so next iteration uses the latest predictions
        prev3 = prev2  # Prev_3 becomes previous Prev_2
        prev2 = prev1  # Prev_2 becomes previous Prev_1
        prev1 = prediction  # Prev_1 becomes the newly predicted value

        # Move forward one year
        current_year += 1  # Increment loop year

    # Return the final predicted crime rate for the target year
    return prediction  # The last computed prediction equals the target year's forecast

# When the user clicks the button, run the forecasting logic
if st.button("Predict Crime Rate"):  # Renders a button and triggers the block when clicked

    # Call the forecasting function with the user's city selection and target year
    result = forecast_crime(  # Stores forecast output (or None)
        city,  # City name selected from dropdown
        target_year,  # Future year chosen by the user
    )  # Ends forecast_crime call

    # If we got a real prediction, show it to the user
    if result is not None:  # Checks if prediction was successful

        # Show a success message containing the predicted crime rate
        st.success(
            f"Predicted Crime Rate for {city} in {target_year}: {result:.2f} crimes per lakh population"  # Formats prediction with 2 decimals
        )  # Ends st.success message

        # Show a large metric widget with just the numeric value
        st.metric(
            label="Predicted Crime Rate",  # Metric label shown in UI
            value=f"{result:.2f}"  # Metric value formatted to 2 decimals
        )  # Ends st.metric configuration

    # If there isn't enough data, show an error to the user
    else:  # Handles the insufficient-history case

        # Show an error message explaining why prediction failed
        st.error(
            "Insufficient historical data available."  # Tells the user prediction couldn't be computed
        )  # Ends st.error message

