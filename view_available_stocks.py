import os
from icecream import ic
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from price_predictor import StockPricePredictor 
import threading

import price_predictor

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
                print(predictor.data.head())

                predictor.prepare_data()
                predictor.build_model()
                # predictor.build_or_load_model("Models/WMT_model.keras")
                history = predictor.train_model(epochs=10, batch_size=32)

                # Ensure the Models directory exists
                if not os.path.exists("Models"):
                    os.makedirs("Models")

                # predictor.save_model(model_path)  # Save the newly trained model
            else:
                raise Exception(f"Failed to fetch data for {ticker}")

            # Predict next day's price
            result = predictor.predict_next_day()
            if result:
                next_price, price_change, percentage_change = result
                # next_price, price_change, percentage_change = (
                #     next_price.item(),
                #     price_change.item(),
                #     percentage_change.item(),
                # )

                #ic(next_price)
                #ic(price_change)
                #ic(percentage_change)

                status_label.configure(text=f"Predicted Price: £{float(next_price):.2f} \n"
                                       f"Change in price: £{float(price_change):.2f} \n"
                                       f"Percentage change: {float(percentage_change):.2f}%")


                messagebox.showinfo(f"Prediction for {ticker} completed",
                                    f"""Predicted Price: £{float(next_price):.2f} 
Change in price: £{float(price_change):.2f}
Percentage change: {float(percentage_change):.2f}%""")

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

