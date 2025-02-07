import os
from icecream import ic
import numpy as np
import tensorflow
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

class StockPricePredictor: # Neural network to predict stock prices declared as a class
    def __init__(self, stock_symbol: str, prediction_days: int = 60): # Constructor
        """
        Initialise the stock price predictor.
        
        Args:
            stock_symbol: Stock ticker symbol (e.g., 'AAPL' for Apple)
            prediction_days: Number of previous days to use for prediction
        """
        self.stock_symbol = stock_symbol
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        self.scaled_data = None
        self.x_train = None
        self.y_train = None
        
    def fetch_data(self, start_date: str = '2022-01-01'): # Fetch stock data from Yahoo Finance
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            start_date: Start date for historical data
        """
        try:
            self.data = yf.download(self.stock_symbol, start=start_date, end=datetime.now())
            if self.data.empty: # Check if data is empty
                print(f"No data fetched for {self.stock_symbol}")
                return False
            print(f"Successfully downloaded data for {self.stock_symbol}")
            return True
        except Exception as e:
            #ic(str(e))
            print(f"Error fetching data: {str(e)}")
            return False

    def prepare_data(self): # Prepare data for LSTM model
        """Prepare data for LSTM model."""
        try:
            if self.data is None or self.data.empty: # Check if data is empty
                print("No data available to prepare.")
                return
            self.scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1)) # Scale the data
            
            x_train = [] # Arrays to store training data
            y_train = [] 

            for x in range(self.prediction_days, len(self.scaled_data)): # Loop through the data and append to arrays
                x_train.append(self.scaled_data[x-self.prediction_days:x, 0])
                y_train.append(self.scaled_data[x, 0])

            self.x_train = np.array(x_train) # Convert to numpy arrays
            self.y_train = np.array(y_train)
            self.x_train = np.reshape(self.x_train, 
                                    (self.x_train.shape[0], self.x_train.shape[1], 1)) # Reshape the data
            print("Training data prepared.")
            return self.data['Close'].iloc[-1] # Return the last closing price
        except Exception as e:
            print(f"Error preparing data for {self.stock_symbol}: {e}")

           
    def build_model(self): # Build and compile the LSTM model
        """Build and compile the LSTM model."""

        self.model = Sequential([ # Sequential model
            LSTM(units=50, return_sequences=True, 
                 input_shape=(self.x_train.shape[1], 1)), # LSTM entry layer with 50 units
            Dropout(0.2), # Dropout layer
            LSTM(units=50, return_sequences=True), # LSTM layer with 50 units
            Dropout(0.2), # Dropout layer
            LSTM(units=50), # LSTM layer with 50 units
            Dropout(0.2), # Dropout layer
            Dense(units=1) # Dense layer with 1 unit for output
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error') # Compile the model
        self.model.fit(self.x_train, self.y_train, epochs=25, batch_size=32) # Fit the model
        
        print("Model built successfully!")

    def train_model(self, epochs: int = 25, batch_size: int = 32): # Train the LSTM model
        """
        Train the LSTM model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        history = self.model.fit(self.x_train, self.y_train, # Train the model using the training data
                               epochs=epochs, batch_size=batch_size, 
                               validation_split=0.1,
                               verbose=1)

        return history
        
    def predict_next_day(self): # Predict the next day's closing price
        """Predict the next day's closing price."""
        try:
            if self.scaled_data is None or len(self.scaled_data) < self.prediction_days: # Check if data is available or sufficient
                print("Insufficient data to make a prediction.")
                return None
            
            last_60_days = self.scaled_data[-self.prediction_days:] # Get the last 60 days of data
            next_day_input = np.reshape(last_60_days, (1, self.prediction_days, 1)) # Reshape the data
            
            if self.model is None: # Check if model is available
                print("Model is not loaded or built.")
                return None
            
            # Make prediction and inverse transform
            prediction = self.model.predict(next_day_input)
            actual_prediction = self.scaler.inverse_transform(prediction)[0][0]
            
            current_price = self.data['Close'].iloc[-1] # Get the current price
            price_change = actual_prediction - current_price
            percentage_change = (price_change / current_price) * 100

            return actual_prediction, price_change, percentage_change # Return the prediction and changes
        except Exception as e: # Error handling
            print(f"Error predicting next day for {self.stock_symbol}: {e}")
