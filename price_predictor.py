import os
from icecream import ic
import numpy as np
import pandas as pd
import pickle
import tensorflow
import yfinance as yf
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta

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
        
    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.model = tensorflow.keras.models.load_model(model_path)
            #ic(self.model)
            print("Loaded existing model.")
            return True
        else:
            print("No existing model found.")
            return False

    def fetch_data(self, start_date: str = '2022-01-01'):
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            start_date: Start date for historical data
        """
        try:
            self.data = yf.download(self.stock_symbol, start=start_date, end=datetime.now())
            #ic(self.data)
            #ic(self.stock_symbol)
            if self.data.empty:
                print(f"No data fetched for {self.stock_symbol}")
                return False
            print(f"Successfully downloaded data for {self.stock_symbol}")
            return True
        except Exception as e:
            #ic(str(e))
            print(f"Error fetching data: {str(e)}")
            return False

    def prepare_data(self):
        """Prepare data for LSTM model."""
        try:
            if self.data is None or self.data.empty:
                print("No data available to prepare.")
                return
            self.scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))
            #ic(self.scaled_data[:5])
            
            x_train = []
            y_train = []

            for x in range(self.prediction_days, len(self.scaled_data)):
                x_train.append(self.scaled_data[x-self.prediction_days:x, 0])
                y_train.append(self.scaled_data[x, 0])

            self.x_train = np.array(x_train)
            self.y_train = np.array(y_train)
            self.x_train = np.reshape(self.x_train, 
                                    (self.x_train.shape[0], self.x_train.shape[1], 1))
            #ic(self.x_train)
            #ic(self.y_train)
            print("Training data prepared.")
        except Exception as e:
            print(f"Error preparing data for {self.stock_symbol}: {e}")

           
    def build_model(self):
        """Build and compile the LSTM model."""
        #ic(self.x_train.shape[1])

        self.model = Sequential([
            LSTM(units=50, return_sequences=True, 
                 input_shape=(self.x_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='mean_squared_error')
        
        print("Model built successfully!")

    def build_or_load_model(self, model_path):
        if not self.load_model(model_path):
            self.build_model()
            print("New model built as no existing models found, training will proceed")
            return False
        print("Model found, no training")
        return True

        
    def train_model(self, epochs: int = 25, batch_size: int = 32):
        """
        Train the LSTM model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        history = self.model.fit(self.x_train, self.y_train, 
                               epochs=epochs, batch_size=batch_size, 
                               validation_split=0.1,
                               verbose=1)
        #ic(history)

        return history
        
    def save_model(self, model_path: str):
        self.model.save(model_path)

    def predict_next_day(self):
        """Predict the next day's closing price."""
        try:
            if self.scaled_data is None or len(self.scaled_data) < self.prediction_days:
                print("Insufficient data to make a prediction.")
                return None
            
            last_60_days = self.scaled_data[-self.prediction_days:]
            next_day_input = np.reshape(last_60_days, (1, self.prediction_days, 1))
            
            if self.model is None:
                print("Model is not loaded or built.")
                return None
            
            # Make prediction and inverse transform
            prediction = self.model.predict(next_day_input)
            actual_prediction = self.scaler.inverse_transform(prediction)[0][0]
            
            current_price = self.data['Close'].iloc[-1]
            price_change = actual_prediction - current_price
            percentage_change = (price_change / current_price) * 100

            #ic(last_60_days)
            #ic(next_day_input)
            #ic(prediction)
            #ic(actual_prediction)
            #ic(current_price)
            #ic(price_change)
            #ic(percentage_change)

            return actual_prediction, price_change, percentage_change
        except Exception as e:
            print(f"Error predicting next day for {self.stock_symbol}: {e}")
           
    
    # def evaluate_model(self):
    #     """evaluate model accuracy using recent predictions."""
    #     # use last 30 days for testing
    #     test_start = len(self.data) - 30
    #     test_data = self.scaled_data[test_start - self.prediction_days:]
        
    #     x_test = []
    #     y_test = self.scaled_data[test_start:].reshape(-1)
        
    #     for x in range(self.prediction_days, len(test_data)):
    #         x_test.append(test_data[x-self.prediction_days:x, 0])
            
    #     x_test = np.array(x_test)
    #     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
    #     predictions = self.model.predict(x_test)
    #     predictions = self.scaler.inverse_transform(predictions)
    #     actual_values = self.scaler.inverse_transform([y_test])
        
    #     # calculate metrics
    #     mse = np.mean((predictions - actual_values.t) ** 2)
    #     rmse = np.sqrt(mse)
    #     mae = np.mean(np.abs(predictions - actual_values.t))
        
    #     return {
    #         'mse': mse,
    #         'rmse': rmse,
    #         'mae': mae,
    #         'predictions': predictions,
    #         'actual_values': actual_values.t
    #     }
    
    # def plot_predictions(self, evaluation_results, save_path='prediction_plot.png'):
    #     """
    #     plot actual vs predicted prices and save to file.
        
    #     args:
    #         evaluation_results (dict): dictionary containing actual_values and predictions
    #         save_path (str): path where to save the plot image file
    #     """
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(evaluation_results['actual_values'], 
    #             label='actual prices', color='blue')
    #     plt.plot(evaluation_results['predictions'], 
    #             label='predicted prices', color='red')
    #     plt.title(f'{self.stock_symbol} stock price prediction results')
    #     plt.xlabel('time (days)')
    #     plt.ylabel('stock price')
    #     plt.legend()
    #     plt.grid(true)
        
    #     # save the plot to file
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()  # close the figure to free memory
       
    # def plot_training_history(self, history):
    #     """plot training history."""
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(history.history['loss'], label='training loss', color='blue')
    #     plt.plot(history.history['val_loss'], label='validation loss', color='red')
    #     plt.title('model training history')
    #     plt.xlabel('epoch')
    #     plt.ylabel('loss')
    #     plt.legend()
    #     plt.grid(true)
    #     plt.show()


# predictor = stockpricepredictor(stock_symbol="aapl")

# # fetch the latest data
# if predictor.fetch_data(start_date='2023-01-01'):  # adjust start_date as needed
#     predictor.prepare_data()
#     predictor.build_model()
#     predictor.train_model()

# # predict when the price will increase
# increase_date, predicted_price, price_change, percentage_change = predictor.predict_increase_date(max_days=60)

# if increase_date:
#     print(f"price is expected to increase on {increase_date.date()} to {predicted_price:.2f}")
#     print(f"change: {price_change:.2f} ({percentage_change:.2f}%)")
# else:
#     print("No price increase expected within the specified future days.")
