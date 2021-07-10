from jqdatasdk import *
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import sleep


def get_factor(stock_code, data_init):
    # 获取特征因子
    MACD = []
    DIFF = []
    DEA = []
    WVAD = []
    OBV = []
    RSI = []
    MA10 = []
    MA20 = []
    MA60 = []
    BOLL_UB = []
    BOLL_MB = []
    BOLL_LB = []
    print(f'获取股票数据:')
    for i in tqdm(range(len(data_init))):
        data_date = data_init.iloc[i].name.date()
        # macd
        diff, dea, macd = technical_analysis.MACD(stock_code, data_date)
        # wvad
        wvad, mawvad = technical_analysis.WVAD(stock_code, data_date)
        # obv
        obv = technical_analysis.OBV(stock_code, data_date)
        # rsi
        rsi = technical_analysis.RSI(stock_code, data_date)
        # ma
        ma10 = technical_analysis.MA(stock_code, data_date, timeperiod=10)
        ma20 = technical_analysis.MA(stock_code, data_date, timeperiod=20)
        ma60 = technical_analysis.MA(stock_code, data_date, timeperiod=60)
        # BOLL
        boll_ub, boll_mb, boll_lb = technical_analysis.Bollinger_Bands(stock_code, data_date)
        # 加入列表
        MACD.append(list(macd.values())[0])
        DIFF.append(list(diff.values())[0])
        DEA.append(list(dea.values())[0])
        WVAD.append(list(wvad.values())[0])
        OBV.append(list(obv.values())[0])
        RSI.append(list(rsi.values())[0])
        MA10.append(list(ma10.values())[0])
        MA20.append(list(ma20.values())[0])
        MA60.append(list(ma60.values())[0])
        BOLL_UB.append(list(boll_ub.values())[0])
        BOLL_MB.append(list(boll_mb.values())[0])
        BOLL_LB.append(list(boll_lb.values())[0])

        sleep(0.01)

    data_init['MACD'] = MACD
    data_init['DIFF'] = DIFF
    data_init['DEA'] = DEA
    data_init['WVAD'] = WVAD
    data_init['OBV'] = OBV
    data_init['RSI'] = RSI
    data_init['MA10'] = MA10
    data_init['MA20'] = MA20
    data_init['MA60'] = MA60
    data_init['BOLL_UB'] = BOLL_UB
    data_init['BOLL_MB'] = BOLL_MB
    data_init['BOLL_LB'] = BOLL_LB

    return data_init
