from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from data.fetch_demand_data import load_demand_csv
import pandas as pd
import numpy as np
from scipy.stats import zscore
from keras.layers import Dropout

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def lstm_forecast(data, column_name):
    series = data[column_name].dropna().values
    series = series.astype('float32').reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    series = scaler.fit_transform(series)
    
    # Split into train and test set
    size = int(len(series) * 0.8)
    train, test = series[0:size], series[size:len(series)]
    
    # Convert to supervised learning problem with a longer time window
    def series_to_supervised(data, n_in=24, n_out=1, dropnan=True):
        n_vars = 1
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg.values
    
    trainX, trainY = series_to_supervised(train)[:,:-1], series_to_supervised(train)[:,-1]
    testX, testY = series_to_supervised(test)[:,:-1], series_to_supervised(test)[:,-1]
    
    # Reshape input to [samples, timesteps, features]
    trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
    
# Define more complex LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=2, shuffle=False, callbacks=[early_stopping])
    
    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    # Invert predictions and actual values to original scale
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY.reshape(-1, 1))
    
    # Calculate MSE
    trainScore = mean_squared_error(trainY, trainPredict)
    testScore = mean_squared_error(testY, testPredict)
    print(f'Train MSE: {trainScore}')
    print(f'Test MSE: {testScore}')
    
    # Calculate RMSE
    trainRMSE = np.sqrt(trainScore)
    testRMSE = np.sqrt(testScore)
    print(f'Train RMSE: {trainRMSE}')
    print(f'Test RMSE: {testRMSE}')


    def forecast_next_24_hours(model, last_data, scaler):
        future_predictions = []
        current_input = last_data.reshape(1, 1, -1)
    
        for _ in range(24):
            prediction = model.predict(current_input)
            future_predictions.append(prediction[0][0])
        
            current_input = np.roll(current_input, -1)
        
            prediction_transformed = scaler.inverse_transform(prediction.reshape(-1, 1))
            current_input[0, -1] = prediction[0][0]
        
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
        return future_predictions.flatten()
    
    # Calculate RMSE
    trainRMSE = np.sqrt(trainScore)
    testRMSE = np.sqrt(testScore)
    print(f'Train RMSE: {trainRMSE}')
    print(f'Test RMSE: {testRMSE}')

    # Forecast next 24 hours
    last_24_hours_data = series[-24:]
    predictions = forecast_next_24_hours(model, last_24_hours_data, scaler)
    print("Predictions for the next 24 hours:")
    print(predictions)

def remove_outliers_using_zscore(data, column_name, threshold=2):
    # Calculate z-scores
    z_scores = zscore(data[column_name])
    abs_z_scores = np.abs(z_scores)
    # Filter entries based on threshold
    filtered_entries = (abs_z_scores < threshold)
    return data[filtered_entries]

def main():
    # Define the path to the CSV file
    csv_path = "../data/raw/ActualForecastReportServlet.csv"  # Adjust the path based on your file structure

    # Load data from the CSV file
    data = load_demand_csv(csv_path)
    print("Loaded CSV data.")
    print(data.head())  # Print data preview
    
    # Remove outliers
    data_cleaned = remove_outliers_using_zscore(data, 'Actual Posted Pool Price')
    print("After removing outliers:")
    print(data_cleaned.head())  # Print cleaned data preview

    # Forecast using LSTM
    lstm_forecast(data_cleaned, 'Actual Posted Pool Price')  # Adjust the column name if necessary


if __name__ == "__main__":
    main()
