import customtkinter as ctk
from cryptography.fernet import Fernet
from encryption_key import load_key
import owned_stocks
from price_predictor import StockPricePredictor
import threading
import logging
import os
import yfinance as yf
from view_available_stocks import view_available_stocks_predictions
from bank_accounts import BankAccountManager
from owned_stocks import OwnedStocksManager
from database_manager import User

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the encryption key
key = load_key()
cipher_suite = Fernet(key)

# Create a class for the stock button
class StockButton(ctk.CTkButton):
    def __init__(self, master, ticker, company_name, **kwargs):
        super().__init__(master, **kwargs)
        self.ticker = ticker
        self.company_name = company_name

def home(current_user_id):
    import customtkinter as ctk  # Importing here to ensure the module is available

    root = ctk.CTk()
    WIDTH, HEIGHT = 1000, 800  # Increased size to accommodate more elements
    root.geometry(f"{WIDTH}x{HEIGHT}")
    root.title("Stock Price Predictor")
    
    view_available_stocks_button = ctk.CTkButton(
        root, 
        text="View available stocks", 
        command=lambda: view_available_stocks_predictions(
            StockButton, logger, homeroot=root, home=home
        )
    )
    view_available_stocks_button.pack(pady=10)

    user = User.get_user_by_id(current_user_id)
    current_username = user.username if user else "Unknown User"

    bank_accounts_button = ctk.CTkButton(
        root, 
        text="Manage bank accounts", 
        command=lambda: BankAccountManager(
            home=home, homeroot=root, current_username=current_username
        )
    )
    bank_accounts_button.pack(pady=10)

    owned_stocks_button = ctk.CTkButton(
        root,
        text="Manage owned stocks",
        command=lambda: OwnedStocksManager(
            home=home, homeroot=root, current_username=current_username
        )
    )
    owned_stocks_button.pack(pady=10)

    root.mainloop()


# if __name__ == "__main__":
    # home()

