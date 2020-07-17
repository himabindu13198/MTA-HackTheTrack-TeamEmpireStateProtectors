    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:
def func_to_run():

    #importing libraries
    import itertools
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm


    # In[2]:


    #reading in data
    data = pd.read_csv('hackathon_data/data.csv')
    # data.head()


    # In[3]:


    #merging date and time
    datetime = data['date'] + ' ' + data['departure_time']


    # In[4]:


    data['datetime'] = pd.to_datetime(datetime)
    #time feature of model


    # In[5]:


    data_updated = data[['datetime','passenger_count']]
    # data_updated.head()


    # In[6]:


    # data_updated.tail()


    # In[7]:


    # from datetime import datetime
    indexed_data = data_updated.set_index(['datetime'])
    # indexed_data.head()
    # list(data['Month'])[-1]


    # In[8]:


    # indexed_data.tail(24*7 + 1)


    # In[9]:


    #printing seasonality in data
    import matplotlib.dates as mdates
    fig = plt.figure(figsize=(20, 2))
    ax = fig.add_subplot(111)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    # plt.xlabel('Date-Time')
    # plt.ylabel('Number of Passengers')
    # plt.plot(indexed_data,linestyle='solid')
    # plt.tight_layout()


    # In[10]:


    # #checking for autocorrelation
    # # autocorrelation
    # print(sm.graphics.tsa.acf(indexed_data, nlags=40))
    # # partial autocorrelation
    # print(sm.graphics.tsa.acf(indexed_data, nlags=40))


    # In[11]:


    # #ACF Plot
    # sm.graphics.tsa.plot_acf(indexed_data, lags=40)
    # plt.show()


    # In[12]:


    # #PACF Plot
    # sm.graphics.tsa.plot_pacf(indexed_data, lags=40)
    # plt.show()


    # In[13]:


    # Define the d and q parameters to take any value between 0 and 1
    q = d = range(0, 2)
    # Define the p parameters to take any value between 0 and 3
    p = range(0, 4)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]

    # print('Examples of parameter combinations for Seasonal ARIMA...')
    # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


    # In[14]:


    train_data = indexed_data['2020-05-22 00:25:00':'2020-05-26 15:56:00']
    test_data = indexed_data['2020-05-26 16:08:00':'2020-05-27 00:12:00']


    # In[15]:


    #grid searching over all possible parameters
    import warnings
    warnings.filterwarnings("ignore") # specify to ignore warning messages

    AIC = []
    SARIMAX_model = []
    counter = 0
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            counter += 1
            if(counter <= 10):
                try:
                    mod = sm.tsa.statespace.SARIMAX(train_data,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    results = mod.fit()

                    # print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
                    AIC.append(results.aic)
                    SARIMAX_model.append([param, param_seasonal])
                except:
                    continue


    # In[16]:


    # print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))


    # In[17]:


    # Let's fit this model with the best params
    mod = sm.tsa.statespace.SARIMAX(train_data,
                                    order=SARIMAX_model[AIC.index(min(AIC))][0],
                                    seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()


    # In[18]:


    #residuals are normal
    # results.plot_diagnostics(figsize=(20, 14))
    # plt.show()


    # In[19]:


    #table to return
    table = []


    # In[48]:


    #predict count of passengers
    pred0 = results.get_prediction(start='2020-05-22 00:25:00', dynamic=False) #monday till friday
    pred0_ci = pred0.conf_int()
    table.append(pred0.predicted_mean[2:8 + 24*7])


    # In[87]:


    #getting the correct hourly predictions for 7 days a week by their index placement
    predictions = []
    for item in table:
        for smallitem in item:
            predictions.append(int(smallitem))
            
    predictions = predictions[len(predictions)//3:][5:173]


    # In[91]:


    # predictions


    # In[94]:


    #convert to dictionary for easy interpretting of day and time
    #getdatetime()
    date_time = ['2020-05-22 00:00:00','2020-05-22 01:00:00','2020-05-22 02:00:00','2020-05-22 03:00:00','2020-05-22 04:00:00','2020-05-22 05:00:00','2020-05-22 06:00:00','2020-05-22 07:00:00','2020-05-22 08:00:00','2020-05-22 09:00:00','2020-05-22 10:00:00','2020-05-22 11:00:00','2020-05-22 12:00:00','2020-05-22 13:00:00','2020-05-22 14:00:00','2020-05-22 15:00:00','2020-05-22 16:00:00','2020-05-22 17:00:00','2020-05-22 18:00:00','2020-05-22 19:00:00','2020-05-22 20:00:00','2020-05-22 21:00:00','2020-05-22 22:00:00','2020-05-22 23:00:00','2020-05-23 00:00:00','2020-05-23 01:00:00','2020-05-23 02:00:00','2020-05-23 03:00:00','2020-05-23 04:00:00','2020-05-23 05:00:00','2020-05-23 06:00:00','2020-05-23 07:00:00','2020-05-23 08:00:00','2020-05-23 09:00:00','2020-05-23 10:00:00','2020-05-23 11:00:00','2020-05-23 12:00:00','2020-05-23 13:00:00','2020-05-23 14:00:00','2020-05-23 15:00:00','2020-05-23 16:00:00','2020-05-23 17:00:00','2020-05-23 18:00:00','2020-05-23 19:00:00','2020-05-23 20:00:00','2020-05-23 21:00:00','2020-05-23 22:00:00','2020-05-23 23:00:00','2020-05-24 00:00:00','2020-05-24 01:00:00','2020-05-24 02:00:00','2020-05-24 03:00:00','2020-05-24 04:00:00','2020-05-24 05:00:00','2020-05-24 06:00:00','2020-05-24 07:00:00','2020-05-24 08:00:00','2020-05-24 09:00:00','2020-05-24 10:00:00','2020-05-24 11:00:00','2020-05-24 12:00:00','2020-05-24 13:00:00','2020-05-24 14:00:00','2020-05-24 15:00:00','2020-05-24 16:00:00','2020-05-24 17:00:00','2020-05-24 18:00:00','2020-05-24 19:00:00','2020-05-24 20:00:00','2020-05-24 21:00:00','2020-05-24 22:00:00','2020-05-24 23:00:00','2020-05-25 00:00:00','2020-05-25 01:00:00','2020-05-25 02:00:00','2020-05-25 03:00:00','2020-05-25 04:00:00','2020-05-25 05:00:00','2020-05-25 06:00:00','2020-05-25 07:00:00','2020-05-25 08:00:00','2020-05-25 09:00:00','2020-05-25 10:00:00','2020-05-25 11:00:00','2020-05-25 12:00:00','2020-05-25 13:00:00','2020-05-25 14:00:00','2020-05-25 15:00:00','2020-05-25 16:00:00','2020-05-25 17:00:00','2020-05-25 18:00:00','2020-05-25 19:00:00','2020-05-25 20:00:00','2020-05-25 21:00:00','2020-05-25 22:00:00','2020-05-25 23:00:00','2020-05-26 00:00:00','2020-05-26 01:00:00','2020-05-26 02:00:00','2020-05-26 03:00:00','2020-05-26 04:00:00','2020-05-26 05:00:00','2020-05-26 06:00:00','2020-05-26 07:00:00','2020-05-26 08:00:00','2020-05-26 09:00:00','2020-05-26 10:00:00','2020-05-26 11:00:00','2020-05-26 12:00:00','2020-05-26 13:00:00','2020-05-26 14:00:00','2020-05-26 15:00:00','2020-05-26 16:00:00','2020-05-26 17:00:00','2020-05-26 18:00:00','2020-05-26 19:00:00','2020-05-26 20:00:00','2020-05-26 21:00:00','2020-05-26 22:00:00','2020-05-26 23:00:00','2020-05-27 00:00:00','2020-05-27 01:00:00','2020-05-27 02:00:00','2020-05-27 03:00:00','2020-05-27 04:00:00','2020-05-27 05:00:00','2020-05-27 06:00:00','2020-05-27 07:00:00','2020-05-27 08:00:00','2020-05-27 09:00:00','2020-05-27 10:00:00','2020-05-27 11:00:00','2020-05-27 12:00:00','2020-05-27 13:00:00','2020-05-27 14:00:00','2020-05-27 15:00:00','2020-05-27 16:00:00','2020-05-27 17:00:00','2020-05-27 18:00:00','2020-05-27 19:00:00','2020-05-27 20:00:00','2020-05-27 21:00:00','2020-05-27 22:00:00','2020-05-27 23:00:00','2020-05-28 00:00:00','2020-05-28 01:00:00','2020-05-28 02:00:00','2020-05-28 03:00:00','2020-05-28 04:00:00','2020-05-28 05:00:00','2020-05-28 06:00:00','2020-05-28 07:00:00','2020-05-28 08:00:00','2020-05-28 09:00:00','2020-05-28 10:00:00','2020-05-28 11:00:00','2020-05-28 12:00:00','2020-05-28 13:00:00','2020-05-28 14:00:00','2020-05-28 15:00:00','2020-05-28 16:00:00','2020-05-28 17:00:00','2020-05-28 18:00:00','2020-05-28 19:00:00','2020-05-28 20:00:00','2020-05-28 21:00:00','2020-05-28 22:00:00','2020-05-28 23:00:00']


    # In[95]:


    prediction_values = dict(zip(date_time, predictions))
    # prediction_values


    # In[96]:


    import json

    with open('data.json', 'w') as fp:
        json.dump(prediction_values, fp)
        
    return(True)

    # In[ ]:


    # pred0.predicted_mean


    # In[ ]:





    # In[ ]:




