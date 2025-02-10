import os
from datetime import datetime
import customtkinter as ctk
from tkinter import messagebox
from price_predictor import StockPricePredictor 
import threading
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import icecream as ic

def view_available_stocks_predictions(StockButton, logger, homeroot, home):
    homeroot.destroy()
    root = ctk.CTk()
    WIDTH = 1000
    HEIGHT = 800
    root.geometry(f"{WIDTH}x{HEIGHT}")
    root.title("Stock Price Predictor")

    AVAILABLE_STOCKS = [
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corporation"),
        ("GOOGL", "Alphabet Inc."),
        ("AMZN", "Amazon.com Inc."),
        ("NVDA", "NVIDIA Corporation"),
        ("META", "Meta Platforms Inc."),
        ("TSLA", "Tesla Inc."),
        ("JPM", "JPMorgan Chase & Co."),
        ("V", "Visa Inc."),
        ("WMT", "Walmart Inc."),
        ("KO", "The Coca-Cola Company"),
        ("DIS", "The Walt Disney Company"),
        ("NFLX", "Netflix Inc."),
        ("ADBE", "Adobe Inc."),
        ("PYPL", "PayPal Holdings Inc."),
        ("INTC", "Intel Corporation"),
        ("AMD", "Advanced Micro Devices Inc."),
        ("CRM", "Salesforce Inc."),
        ("BA", "Boeing Company"),
        ("GE", "General Electric Company")
    ]

    def process_stock(ticker, results_frame):
        try:
            stock_frame = ctk.CTkFrame(results_frame)
            stock_frame.pack(pady=5, padx=10, fill="x")

            status_label = ctk.CTkLabel(stock_frame, text=f"Fetching data for {ticker}...")
            status_label.pack(pady=5)

            predictor = StockPricePredictor(ticker)

            status_label.configure(text=f"Training new model for {ticker}...")
            if predictor.fetch_data():
                # print(predictor.data.head())
                predictor.prepare_data()
                predictor.build_model()

            # Predict next day's price
            result = predictor.predict_next_day()
            if result:
                next_price, price_change, percentage_change = result
                stock = yf.Ticker(ticker)
                market_cap = stock.info['marketCap']

                current_price = stock.history(period="1d")['Close'].iloc[-1]
                num_shares = market_cap / current_price

                price_history = stock.history(period="1y")['Close']

                # Display graph of historical prices
                fig, ax = plt.subplots()
                ax.plot(price_history, label='Historical Prices')
                ax.plot(price_history.index[-1], next_price, 'ro', label='Predicted Price')
                plt.xticks = (0,80)
                ax.set_title(f"{ticker} Historical Prices and Prediction")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend()

                canvas = FigureCanvasTkAgg(fig, master=stock_frame)
                canvas.draw()
                canvas.get_tk_widget().pack()

                status_label.configure(text=f"Current Price: £{float(current_price):.2f} \n"
                                       f"Predicted Price: £{float(next_price):.2f} \n"
                                       f"Change in price: £{float(price_change.iloc[0]):.2f} \n"
                                       f"Percentage change: {float(percentage_change.iloc[0]):.2f}% \n"
                                       f"Number of available shares: {num_shares:.2f}")
                                       # f"Number of available shares: {get_number_of_available_shares(ticker)}")


                messagebox.showinfo(f"Prediction for {ticker} completed",
                                    f"""Predicted Price: £{float(next_price):.2f} 
Change in price: £{float(price_change.iloc[0]):.2f}
Percentage change: {float(percentage_change.iloc[0]):.2f}%""")

            elif predictor.fetch_data() == False:
                raise Exception(f"Failed to fetch data for {ticker}")

            else:
                print("Prediction failed")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            status_label.configure(text=f"Error: {str(e)}")

    def display_stock_prediction(ticker, company_name):
        for widget in results_frame.winfo_children():
            widget.destroy()

        header = ctk.CTkLabel(results_frame, text=f"Processing {ticker} ({company_name})", font=("Arial", 20, "bold"))
        header.pack(pady=10)

        info_label = ctk.CTkLabel(results_frame, text="Fetching data... Please wait.", font=("Arial", 12))
        info_label.pack(pady=5)

        thread = threading.Thread(target=process_stock, args=(ticker, results_frame), daemon=True)
        thread.start()

    def return_home(home):
        with open("user_id.txt", "r") as f:
            lines = f.readlines()
            # current_user_id = lines[0].strip()
            current_username = lines[1].strip()

        root.destroy()
        home(current_username)

    left_panel = ctk.CTkFrame(root, width=300)
    left_panel.pack(side="left", fill="y", padx=10, pady=10)
    left_panel.pack_propagate(False)

    return_home_button = ctk.CTkButton(left_panel, text="Return Home", command=lambda: return_home(home=home))
    return_home_button.pack(pady=10)

    stock_scroll = ctk.CTkScrollableFrame(left_panel)
    stock_scroll.pack(fill="both", expand=True, padx=5, pady=5)

    for ticker, company in AVAILABLE_STOCKS:
        button = ctk.CTkButton(
            stock_scroll,
            text=f"{ticker}\n{company}",
            command=lambda t=ticker, c=company: display_stock_prediction(t, c),
            font=("Arial", 14),
            height=60
        )
        button.pack(pady=2, padx=5, fill="x")

    right_panel = ctk.CTkFrame(root)
    right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    welcome_label = ctk.CTkLabel(right_panel, text="Welcome to Forecastr!", font=("Arial", 24, "bold"))
    welcome_label.pack(pady=10)

    instruction_label = ctk.CTkLabel(right_panel, text="Select a stock to view its prediction.", font=("Arial", 14))
    instruction_label.pack(pady=5)

    results_frame = ctk.CTkFrame(right_panel)
    results_frame.pack(fill="both", expand=True, padx=10, pady=10)

    root.mainloop()
