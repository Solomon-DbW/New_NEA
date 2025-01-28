#!/usr/bin/env python
# coding: utf-8

# # Downloading S&P 500 price data

# In[1]:


import yfinance as yf


# In[2]:


sp500 = yf.Ticker("^GSPC")


# In[3]:


sp500 = sp500.history(period="max")


# In[4]:


sp500


# In[5]:


sp500.index


# # Cleaning and visualising Stock Market data

# In[6]:


sp500.plot.line(y="Close", use_index=True) # Plots 'trading days' against their corresponding closing prices


# In[7]:


del sp500["Dividends"]
del sp500["Stock Splits"] # These columns are more useful for individual stocks, not an index


# # Setting up Target for Machine Learning

# In[8]:


sp500["Tomorrow"] = sp500["Close"].shift(-1) # Column for next day's closing price


# In[9]:


sp500


# In[10]:


sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int) # If tomorrow's closing price is greater than today's, astype(int) makes boolean a number where 1 means it's greater and 0 means it isn't


# In[11]:


sp500


# In[12]:


sp500 = sp500.loc["1990-01-01":].copy() # Only rows with indexes after 01/01/1990 as previous data is irrelevant due to major changes in the market
# .copy() needed to prevent pandas 'setting with copy' warning when trying to subset a dataframe and later assigning back to it.


# In[13]:


sp500


# # Training an Initial Machine Learning Model

# In[14]:


from sklearn.ensemble import RandomForestClassifier
# Random Forest chosen because:
#    1. It trains multiple decision trees with randomised parameters and averaging the results from the trees, which makes Random Forests more resistant to overfitting than other models
#    2. They run relatively quickly
#    3. Can detect non-linear tendencies in the data

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)   # n_estimators = n_individual decision trees, min_sample_split helps protect against overfitting but the higher it is, the lower the accuracy, random_state causes random numbers that are generated to be in a predictable sequence each time which helps when updating or improving model

train = sp500.iloc[:-100] # Data is time-series so cross validation is unsuitable as it produces good results when training but terrible in real life due to the fact that it doesn't take the data's time-series nature into account, which causes future data to be used to predict the past and is basically giving the model clues it won't get in real-life (leaking)
test = sp500.iloc[-100:] 

predictors = ["Close", "Volume", "Open", "High", "Low"] # Columns used to make predictions
model.fit(train[predictors], train["Target"]) # Uses 'predictors' columns to train the model in order to predict 'Target'


# # Measuring model accuracy

# In[15]:


from sklearn.metrics import precision_score
import pandas as pd

preds = model.predict(test[predictors]) # Generates predictions using model using 'test' dataset


# In[16]:


preds = pd.Series(preds, index=test.index)


# In[17]:


precision_score(test["Target"], preds) # Calculate precision score using actual Target and predicted Target


# In[18]:


combined = pd.concat([test["Target"], preds], axis=1) # Concatenates/combines actual Target values with predicted Target values in order to plot the predictions, axis=1 means each input is treated as a column


# In[19]:


combined.plot() # 0(orange) = predictions, Target(blue) = actual values


# # Building a Backtesting System

# In[20]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[21]:


def backtest(data, model, predictors, start=2500, step=250): # 1 trading year ~ 250 days so start=2500 takes 10 years of data as training data for model, step=250 trains model on each year's data one at a time so it can predict for multiple years
    all_predictions = []

    for i in range(start, data.shape[0], step): # Iterates across data year by year and make predictions for all years apart from the first 10
        train = data.iloc[0:i].copy() # All years before current year
        test = data.iloc[i:(i+step)].copy() # Current year
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions) # Appends predictions for given year

    return pd.concat(all_predictions)


# In[22]:


predictions = backtest(sp500, model, predictors)


# In[23]:


predictions["Predictions"].value_counts()


# In[24]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[25]:


predictions["Target"].value_counts() / predictions.shape[0] # Benchmark for accuracy by dividing counts for '0' and '1' by the number of rows


# # Adding Additional Predictors to Model

# In[26]:


horizons = [2, 5, 60, 250,1000] # Horizons on which we want to look at rolling means: Mean 'Close' price for last 2 days, last 5 days(week), last 60 days(3 months), last year and last 4 years and find the mean of today's 'Close' price and the 'Close' prices in the horizon periods
new_predictors = [] # Holds new columns

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean() # Calculate rolling average against horizon and take the mean

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"] # S&P500 close price divided by calculated rolling average

    trend_column = f"Trend_{horizon}" # Number of days in past {horizon} days where stock price went up
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"] # Forward shift and rolling sum of target, shows 'NaN' if pandas can't find enough days/rows before current row to calculate rolling average

    new_predictors += [ratio_column, trend_column]


# In[27]:


sp500.dropna()


# # Improving Model

# In[34]:


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[35]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1] # Returns a probability that the row will be '0' or '1' instead of '0' or '1'  to give more control over what becomes a '0' or '1', [:,1} returns second column which is probability that stock price goes up
    #preds[preds >= 0.6] = 1 # Set custom threshold where a probability >= 60% results in the model returning that the price will increase, which reduces total number of days it predicts a price increase but increase the chance that the price will actually go up on those days
    #preds[preds < 0.6] = 0
    preds[preds >= 0.7] = 1
    preds[preds < 0.7] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[36]:


predictors = backtest(sp500, model, new_predictors) # No Close, open, high, low and volume as they're just numbers that aren't actually informative for the model, but the ratios are most informative


# In[37]:


predictions["Predictions"].value_counts()


# In[32]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[ ]:




