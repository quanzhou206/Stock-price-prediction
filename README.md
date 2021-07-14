# Stock-price-prediction
This project uses jqdata to forecast the price of Chinese stock.  The methods used include LSTM, LSTM_CNN, CNN_ LSTM, AdaBoost, random forest, and using AdaBoost to integrate LSTM
## Usage
1. Register a jqdata account:https://www.joinquant.com/
2. Modify your jqdata account in line 16 of main.py
3. run main.py, the data of CSI300 will be used by default.
## Attention
1. The network models contained in the folder are trained based on CSI300. When the stock code is changed, the model should be deleted and the main.py training should be rerun.
## Result
### LSTM result
![LSTM](https://github.com/quanzhou206/image_res/blob/main/LSTM.png?raw=true)
### CNN_LSTM result
![CNN_LSTM](https://github.com/quanzhou206/image_res/blob/main/CNN_LSTM.png?raw=true)
### LSTM_CNN result
![LSTM_CNN](https://github.com/quanzhou206/image_res/blob/main/LSTM_CNN.png?raw=true)
### Adaboost result
![Adaboost](https://github.com/quanzhou206/image_res/blob/main/Adaboost.png?raw=true)
### Random_forest result
![Random_forest](https://github.com/quanzhou206/image_res/blob/main/RF.png?raw=true)
### Ridge integrate result
![Ridge integrate](https://github.com/quanzhou206/image_res/blob/main/Ridge.png?raw=true)
### Adaboost_LSTM result
![Adaboost_LSTM](https://github.com/quanzhou206/image_res/blob/main/LSTM_ADA.png?raw=true)
