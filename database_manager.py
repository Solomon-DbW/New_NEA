from typing import Text
from sqlalchemy import DateTime, False_, Float, create_engine, Column, Integer, String, ForeignKey, MetaData
from sqlalchemy.orm.decl_api import DeclarativeBase
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
import sqlite3

def migrate_database():
    # 1. First backup existing data
    conn = sqlite3.connect('users_and_details.db')
    cursor = conn.cursor()
    
    # Get existing users
    cursor.execute("SELECT username, password FROM users")
    existing_users = cursor.fetchall()
    
    # Try to get existing cards if table exists
    try:
        cursor.execute("SELECT userid, card_holder_name, card_number, expiration_date, card_type, cvv_code FROM cards")
        existing_cards = cursor.fetchall()
    except sqlite3.OperationalError:
        existing_cards = []
    
    # 2. Drop existing tables
    cursor.execute("DROP TABLE IF EXISTS cards")
    cursor.execute("DROP TABLE IF EXISTS users")
    
    # 3. Create new tables with correct schema
    cursor.execute("""
        CREATE TABLE users (
            userid INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR UNIQUE,
            password VARCHAR
        )
    """)
    
    cursor.execute("""
        CREATE TABLE cards (
            cardid INTEGER PRIMARY KEY AUTOINCREMENT,
            userid INTEGER,
            card_holder_name VARCHAR,
            card_number VARCHAR,
            expiration_date VARCHAR,
            card_type VARCHAR,
            cvv_code VARCHAR,
            FOREIGN KEY (userid) REFERENCES users(userid) ON DELETE CASCADE
        )
    """)
    
    # 4. Reinsert the data
    cursor.executemany(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        existing_users
    )
    
    # Get the new userids for existing users
    user_mapping = {}
    for username, _ in existing_users:
        cursor.execute("SELECT userid FROM users WHERE username = ?", (username,))
        userid = cursor.fetchone()[0]
        user_mapping[username] = userid
    
    # Reinsert cards with updated userids if there were any
    if existing_cards:
        cursor.executemany(
            """INSERT INTO cards 
               (userid, card_holder_name, card_number, expiration_date, card_type, cvv_code) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            existing_cards
        )
    
    conn.commit()
    conn.close()

# Initialize SQLAlchemy
engine = create_engine("sqlite:///users_and_details.db")
Session = sessionmaker(bind=engine)
session = Session()

# Base class definition
class Base(DeclarativeBase):
    pass

# User table model
class User(Base):
    __tablename__ = "users"
    userid = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True)
    password = Column(String)
    
    # Relationship to cards
    cards = relationship("Card", backref="user", cascade="all, delete-orphan")
    
    def __init__(self, username, password): # Constructor
        self.username = username
        self.password = password

    @staticmethod
    def get_username(user_id: int): # Static method to get username by user ID
        """Retrieve username by user ID."""
        try:
            user = session.query(User).filter_by(userid=user_id).first() # Query the user by user ID
            return user.username if user else None
        except SQLAlchemyError as e:
            print(f"Error retrieving username: {str(e)}")
            return None


    def get_user_id(self): # Method to get user ID
        return self.userid

    @staticmethod
    def get_user_by_username(username: str): # Static method to get user by username
        """
        Retrieve a user by their username.
        Returns None if user not found or if there's an error.
        """
        try:
            user = session.query(User).filter_by(username=username).first()
            return user
        except SQLAlchemyError as e:
            print(f"Error retrieving user: {str(e)}")
            return None

    @staticmethod
    def get_user_by_id(user_id: int): # Static method to get user by user ID 
        """Retrieve a user by their ID. Returns None if user not found or if there's an error."""
        try:
            user = session.query(User).filter_by(userid=user_id).first()
            return user
        except SQLAlchemyError as e:
            print(f"Error retrieving user by ID: {str(e)}")
            return None

# Card table model
class Card(Base):
    __tablename__ = "cards"
    cardid = Column(Integer, primary_key=True, autoincrement=True)
    userid = Column(Integer, ForeignKey("users.userid", ondelete="CASCADE"))
    card_holder_name = Column(String)
    card_number = Column(String)
    expiration_date = Column(String)
    card_type = Column(String)
    cvv_code = Column(String)
    
    def __init__(self, userid, card_holder_name, card_number, expiration_date, card_type, cvv_code): # Constructor
        self.userid = userid
        self.card_holder_name = card_holder_name
        self.card_number = card_number
        self.expiration_date = expiration_date
        self.card_type = card_type
        self.cvv_code = cvv_code
    
    def __repr__(self): # Representation of the card
        return f"Card(cardid={self.cardid}, userid={self.userid}, card_holder_name={self.card_holder_name})"
    
    def save_card(self): # Method to save the card
        """Add this card to the session and commit."""
        try:
            session.add(self)
            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error saving card: {str(e)}")
            return False
    
    @staticmethod
    def delete_card(card_id: int) -> bool: # Static method to delete a card by ID
        """Delete a card by ID."""
        try:
            card = session.query(Card).filter_by(cardid=card_id).first() # Query the card by card ID
            if card:
                session.delete(card)
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error deleting card: {str(e)}")
            return False

class OwnedStock(Base): # Owned stocks table model
    __tablename__ = "Owned_Stocks_and_Investments"
    userid = Column(Integer, ForeignKey("users.userid", ondelete="CASCADE"))
    stockid = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    stock_ticker = Column(String)
    date_purchased = Column(String)
    amount_invested = Column(Float)
    number_of_shares = Column(Integer)

    def save_stock(self): # Method to save the stock
        try:
            session.add(self)
            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error saving stock: {str(e)}")
            return False

    @staticmethod
    def delete_stock(stock_id: int) -> bool: # Static method to delete a stock by ID
        try:
            stock = session.query(OwnedStock).filter_by(stockid=stock_id).first()
            if stock:
                session.delete(stock)
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error deleting stock:{str(e)}")
            return False

    @staticmethod
    def get_owned_stock_by_user_id_and_ticker(user_id: int, ticker: str): # Static method to get owned stock by user ID and ticker
        try:
            stock = session.query(OwnedStock).filter_by(userid=user_id, stock_ticker=ticker).first() # Query the stock by user ID and ticker
            return stock
        except SQLAlchemyError as e:
            print(f"Error retrieving stock: {str(e)}")
            return None

    @staticmethod
    def get_owned_stock_price_by_user_id_and_ticker(user_id: int, ticker: str): # Static method to get owned stock price by user ID and ticker
        try:
            stock = session.query(OwnedStock).filter_by(userid=user_id, stock_ticker=ticker).first() # Query the stock by user ID and ticker
            if stock is not None:
                return stock.amount_invested
            return None
        except SQLAlchemyError as e:
            print(f"Error retrieving stock price: {str(e)}")
            return None



# Create all tables
Base.metadata.create_all(engine)
