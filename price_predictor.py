import os
from icecream import ic
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime

class StockPricePredictor:
    def __init__(self, stock_symbol: str, prediction_days: int = 60):
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
        

    def save_model(self, model_path: str):
        """
        Save the model to the specified path.
        
        Args:
            model_path: Path to save the model directory (without extension)
        """
        if self.model is not None:
            # Append .keras extension to the model path
            model_path_with_extension = f"Models/{model_path}.keras"
            self.model.save(model_path_with_extension)
            print(f"Model saved to {model_path_with_extension}")
        else:
            print("No model to save.")

    def load_model(self, model_path: str):
        """
        Load a pre-trained model from the specified path.
        
        Args:
            model_path: Path to the saved model directory (without extension)
        """
        model_path_with_extension = f"Models/{model_path}.keras"
        if os.path.exists(model_path_with_extension):
            self.model = tf.keras.models.load_model(model_path_with_extension)
            print("Loaded existing model.")
            return True
        else:
            print("No existing model found.")
            return False

    def fetch_data(self, start_date: str = '2020-01-01'):
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            start_date: Start date for historical data
        """
        try:
            self.data = yf.download(self.stock_symbol, start=start_date, end=datetime.now())
            if self.data.empty:
                print(f"No data fetched for {self.stock_symbol}")
                return False
            print(f"Successfully downloaded data for {self.stock_symbol}")
            return True
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return False

    def prepare_data(self):
        """Prepare data for LSTM model."""
        try:
            if self.data is None or self.data.empty:
                print("No data available to prepare.")
                return
            
            # Use multiple features (Open, High, Low, Close, Volume)
            features = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
            self.scaled_data = self.scaler.fit_transform(features)
            
            x_train = []
            y_train = []

            for x in range(self.prediction_days, len(self.scaled_data)):
                x_train.append(self.scaled_data[x-self.prediction_days:x, :])
                y_train.append(self.scaled_data[x, 3])  # 3 is the index of 'Close'

            self.x_train = np.array(x_train)
            self.y_train = np.array(y_train)
            print("Training data prepared.")
        except Exception as e:
            print(f"Error preparing data for {self.stock_symbol}: {e}")

    def build_model(self):
        """Build and compile the LSTM model."""
        self.model = Sequential([
            Bidirectional(LSTM(units=100, return_sequences=True, 
                              input_shape=(self.x_train.shape[1], self.x_train.shape[2]))),
            Dropout(0.3),
            Bidirectional(LSTM(units=100, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(units=100)),
            Dropout(0.3),
            Dense(units=50, activation='relu'),
            Dense(units=1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='mean_squared_error')
        
        print("Model built successfully!")

    def train_model(self, epochs: int = 50, batch_size: int = 32):
        """
        Train the LSTM model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        history = self.model.fit(self.x_train, self.y_train, 
                               epochs=epochs, batch_size=batch_size, 
                               validation_split=0.2,
                               callbacks=[early_stopping, reduce_lr],
                               verbose=1)
        return history
        
    def predict_next_day(self):
        """Predict the next day's closing price."""
        try:
            if self.scaled_data is None or len(self.scaled_data) < self.prediction_days:
                print("Insufficient data to make a prediction.")
                return None
            
            last_60_days = self.scaled_data[-self.prediction_days:]
            next_day_input = np.reshape(last_60_days, (1, self.prediction_days, self.scaled_data.shape[1]))
            
            if self.model is None:
                print("Model is not loaded or built.")
                return None
            
            # Make prediction and inverse transform
            prediction = self.model.predict(next_day_input)
            actual_prediction = self.scaler.inverse_transform(
                np.concatenate((last_60_days[-1, :-1], prediction), axis=None).reshape(1, -1)
            )[0][-1]
            
            current_price = self.data['Close'].iloc[-1]
            price_change = actual_prediction - current_price
            percentage_change = (price_change / current_price) * 100

            return actual_prediction, price_change, percentage_change
        except Exception as e:
            print(f"Error predicting next day for {self.stock_symbol}: {e}")
            return None

    def evaluate_model(self, test_days: int = 30):
        """
        Evaluate the model's performance by comparing predictions with actual data.
        
        Args:
            test_days: Number of days to use for testing (default: 30)
        
        Returns:
            A dictionary containing evaluation metrics and data for plotting.
        """
        try:
            if self.scaled_data is None or len(self.scaled_data) < self.prediction_days + test_days:
                print("Insufficient data for evaluation.")
                return None

            # Prepare test data
            test_start = len(self.scaled_data) - test_days
            test_data = self.scaled_data[test_start - self.prediction_days:]

            x_test = []
            y_test = self.scaled_data[test_start:, 3]  # 3 is the index of 'Close'

            for x in range(self.prediction_days, len(test_data)):
                x_test.append(test_data[x - self.prediction_days:x, :])

            x_test = np.array(x_test)
            y_test = np.array(y_test)

            # Generate predictions
            predictions = self.model.predict(x_test)
            predictions = predictions.reshape(-1, 1)  # Reshape predictions to match y_test shape

            # Inverse transform predictions and y_test
            # Create a dummy array to match the original feature dimensions
            dummy_features = np.zeros((len(predictions), self.scaled_data.shape[1]))
            dummy_features[:, 3] = predictions.flatten()  # Replace 'Close' column with predictions
            predictions = self.scaler.inverse_transform(dummy_features)[:, 3]

            dummy_features[:, 3] = y_test  # Replace 'Close' column with actual values
            y_test = self.scaler.inverse_transform(dummy_features)[:, 3]

            # Calculate evaluation metrics
            mse = np.mean((predictions - y_test) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y_test))

            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'predictions': predictions,
                'actual_values': y_test
            }
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return None

# Example usage:
# predictor = StockPricePredictor('AAPL')
# predictor.fetch_data()
# predictor.prepare_data()
# predictor.build_model()
# predictor.train_model()
# predictor.save_model('my_model')
# prediction = predictor.predict_next_day()
# print(prediction)
