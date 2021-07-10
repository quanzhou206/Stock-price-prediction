# Stock-price-prediction
This project uses jqdata to forecast the price of Chinese stock.  The methods used include LSTM, LSTM_CNN, CNN_ LSTM, AdaBoost, random forest, and using AdaBoost to integrate LSTM
## Usage
1. Register a jqdata account:https://www.joinquant.com/
2. Modify your jqdata account in line 16 of main.py
3. run main.py, the data of CSI300 will be used by default.
## Attention
1. Running main.py for the first time will get stock data from jqdata and save the data to local. The second run main.py will call each model for stock price forecast.
2. The network models contained in the folder are trained based on CSI300. When the stock code is changed, the model should be deleted and the main.py training should be rerun.
