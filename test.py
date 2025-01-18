import yfinance as yf

# Define the stock ticker
ticker = "AAPL"

# Create a Ticker object
stock = yf.Ticker(ticker)

# Retrieve the market capitalization and current stock price
market_cap = stock.info['marketCap']
current_price = stock.info['currentPrice']

# Calculate the number of shares
num_shares = market_cap / current_price
data = stock.history(period="1y")  # Fetches data for the last year
print(data)
print(f"The number of available shares for {ticker} is approximately {num_shares:.2f}")
