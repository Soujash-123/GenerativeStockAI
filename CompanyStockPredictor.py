import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from google.colab import drive
import numpy as np
from math import sqrt

# Mount Google Drive
drive.mount('/content/drive')

# Define the file path in your Google Drive
file_path = '/content/drive/My Drive/stock_data_filtered.csv'

# Load the dataset from Google Drive
df = pd.read_csv(file_path)

def preprocess_data(df):
    # Find rows with missing values
    missing_rows = df[df.isnull().any(axis=1)]

    if not missing_rows.empty:
        print("Rows with missing values:")
        print(missing_rows)

    # Preprocess the dataset: Replace NaN values with column means for numeric columns only
    numeric_columns = ['Open', 'High', 'Low', 'Volume', 'Close']
    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Add a datetime column for easy filtering
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])

    return df

def step1_prediction(df, date_to_predict):
    # Define the features (X) and target variable (y) for the first step of prediction
    X_step1 = df[['Open']]
    y_step1 = df[['High', 'Low', 'Volume']]

    # Specify the date to predict
    date_to_predict = pd.to_datetime(date_to_predict)

    # Define the time range from 9:30 AM to 4:00 PM
    start_time = pd.Timestamp(f"{date_to_predict.date()} 09:30:00")
    end_time = pd.Timestamp(f"{date_to_predict.date()} 16:00:00")

    # Create an empty DataFrame to store predictions for the first step
    predictions_step1_df = pd.DataFrame(columns=['Date and Time', 'Predicted Open', 'Predicted High', 'Predicted Low', 'Predicted Volume'])

    # Loop through the minutes within the specified time frame
    current_time = start_time
    while current_time <= end_time:
        # Filter data for the current minute
        filtered_data = df[(df['Date and Time'] == current_time)]

        # If there is no data available for the current minute, continue predicting using the last available data
        if filtered_data.empty:
            # Use the last available data point for prediction
            last_available_data = df[df['Date and Time'] < current_time].iloc[-1]

            # Create synthetic data by copying the last available data
            synthetic_data = {
                'Date and Time': [current_time],
                'Open': [last_available_data['Open']]
            }

            # Append the synthetic data to the predictions DataFrame
            predictions_step1_df = pd.concat([predictions_step1_df, pd.DataFrame(synthetic_data)], ignore_index=True)
        else:
            # Split the filtered dataset into features (X) for prediction
            X_filtered_step1 = filtered_data[['Open']]

            # Create and train a simple Linear Regression model for step 1
            model_step1 = LinearRegression()
            model_step1.fit(X_step1, y_step1)

            # Make predictions for the filtered data
            y_pred_step1 = model_step1.predict(X_filtered_step1)

            # Append the predictions to the predictions DataFrame for step 1
            predictions_step1_df = pd.concat([predictions_step1_df, pd.DataFrame({
                'Date and Time': [current_time],
                'Predicted Open': [X_filtered_step1['Open'].values[0]],
                'Predicted High': [y_pred_step1[0][0]],
                'Predicted Low': [y_pred_step1[0][1]],
                'Predicted Volume': [y_pred_step1[0][2]]
            })], ignore_index=True)

        # Increment the current time by 1 minute
        current_time += pd.Timedelta(minutes=1)

    return predictions_step1_df

def step2_prediction(df, predictions_step1_df, date_to_predict):
    # Define the features (X) and target variable (y) for the second step of prediction
    X_step2 = predictions_step1_df[['Predicted Open', 'Predicted High', 'Predicted Low', 'Predicted Volume']]
    y_step2 = df[df['Date and Time'].dt.date == pd.to_datetime(date_to_predict).date()]['Close']

    # Specify the date to predict
    date_to_predict = pd.to_datetime(date_to_predict)

    # Define the time range from 9:30 AM to 4:00 PM
    start_time = pd.Timestamp(f"{date_to_predict.date()} 09:30:00")
    end_time = pd.Timestamp(f"{date_to_predict.date()} 16:00:00")

    # Create an empty DataFrame to store predictions for the second step
    predictions_step2_df = pd.DataFrame(columns=['Date and Time', 'Predicted Close'])

    # Loop through the minutes within the specified time frame
    current_time = start_time
    while current_time <= end_time:
        # Filter data for the current minute
        filtered_data = predictions_step1_df[(predictions_step1_df['Date and Time'] == current_time)]

        # If there is no data available for the current minute, continue predicting using the last available data
        if filtered_data.empty:
            # Use the last available data point for prediction
            last_available_data = predictions_step1_df[predictions_step1_df['Date and Time'] < current_time].iloc[-1]

            # Create synthetic data by copying the last available data
            synthetic_data = {
                'Date and Time': [current_time],
                'Predicted Open': [last_available_data['Predicted Open']],
                'Predicted High': [last_available_data['Predicted High']],
                'Predicted Low': [last_available_data['Predicted Low']],
                'Predicted Volume': [last_available_data['Predicted Volume']]
            }

            # Append the synthetic data to the predictions DataFrame for step 2
            predictions_step2_df = pd.concat([predictions_step2_df, pd.DataFrame(synthetic_data)], ignore_index=True)
        else:
            # Split the filtered dataset into features (X) for prediction
            X_filtered_step2 = filtered_data[['Predicted Open', 'Predicted High', 'Predicted Low', 'Predicted Volume']]

            # Create and train a simple Linear Regression model for step 2
            model_step2 = LinearRegression()
            model_step2.fit(X_step2, y_step2)

            # Make predictions for the filtered data
            y_pred_step2 = model_step2.predict(X_filtered_step2)

            # Append the predictions to the predictions DataFrame for step 2
            predictions_step2_df = pd.concat([predictions_step2_df, pd.DataFrame({
                'Date and Time': [current_time],
                'Predicted Close': [y_pred_step2[0]]
            })], ignore_index=True)

        # Increment the current time by 1 minute
        current_time += pd.Timedelta(minutes=1)

    return predictions_step2_df

# Preprocess the data
df = preprocess_data(df)

# Specify the date to predict
date_to_predict = '2023-09-15'

# Perform Step 1 prediction
predictions_step1_df = step1_prediction(df, date_to_predict)

# Perform Step 2 prediction
predictions_step2_df = step2_prediction(df, predictions_step1_df, date_to_predict)

# Calculate and print the Root Mean Squared Error (RMSE) for step 2
rmse = sqrt(mean_squared_error(df[df['Date and Time'].dt.date == pd.to_datetime(date_to_predict).date()]['Close'], predictions_step2_df['Predicted Close']))
print(f'RMSE for step 2: {rmse}')

# Print the data for both steps
print("Predictions for Step 1:")
print(predictions_step1_df)
print("\nPredictions for Step 2:")
print(predictions_step2_df)
