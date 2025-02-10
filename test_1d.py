from price_predictor import StockPricePredictor

AVAILABLE_STOCKS = [ # List of available stocks
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

for stock in AVAILABLE_STOCKS:
    test_predictor = StockPricePredictor(stock[0])
    test_predictor.fetch_data()
    test_predictor.prepare_data()
    with open("Training_arrays.txt", "a") as f:
        f.write(f"{stock[0]} x_train = {test_predictor.x_train} \n {stock[0]} y_train = {test_predictor.y_train}") 
