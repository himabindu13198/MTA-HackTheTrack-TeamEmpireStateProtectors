#!/usr/bin/env python
# coding: utf-8

# In[1]:

def func_to_run:
	#importing libraries
	import itertools
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	import statsmodels.api as sm


# In[3]:


#reading in data
	data = pd.read_csv('hackathon_data/data.csv')
	data.head()


# In[4]:


#merging date and time
	datetime = data['date'] + ' ' + data['departure_time']


# In[5]:


	data['datetime'] = pd.to_datetime(datetime)
#time feature of model


# In[6]:


	data_updated = data[['datetime','passenger_count']]
	data_updated.head()


# In[7]:


	data_updated.tail()


# In[8]:


# from datetime import datetime
	indexed_data = data_updated.set_index(['datetime'])
	indexed_data.head()
# list(data['Month'])[-1]


# In[9]:


	indexed_data.tail(24*7 + 1)


# In[10]:


#printing seasonality in data
	import matplotlib.dates as mdates
	fig = plt.figure(figsize=(20, 2))
	ax = fig.add_subplot(111)
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
	plt.xlabel('Date-Time')
	plt.ylabel('Number of Passengers')
	plt.plot(indexed_data,linestyle='solid')
	plt.tight_layout()


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


# In[11]:


# Define the d and q parameters to take any value between 0 and 1
	q = d = range(0, 2)
# Define the p parameters to take any value between 0 and 3
	p = range(0, 4)

# Generate all different combinations of p, q and q triplets
	pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
	seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]

	print('Examples of parameter combinations for Seasonal ARIMA...')
	print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
	print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
	print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
	print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[12]:


	train_data = indexed_data['2020-05-22 00:25:00':'2020-05-26 15:56:00']
	test_data = indexed_data['2020-05-26 16:08:00':'2020-05-27 00:12:00']


# In[13]:


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

               	 	print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
                	AIC.append(results.aic)
                	SARIMAX_model.append([param, param_seasonal])
            	except:
                	continue


# In[14]:


	print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))


# In[15]:


# Let's fit this model with the best params
	mod = sm.tsa.statespace.SARIMAX(train_data,
                                order=SARIMAX_model[AIC.index(min(AIC))][0],
                                seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

	results = mod.fit()


# In[16]:


#residuals are normal
	results.plot_diagnostics(figsize=(20, 14))
	plt.show()


# In[ ]:


#table to return
	table = []


# In[ ]:


#predict count of passengers
	pred0 = results.get_prediction(start='2020-05-18 00:00:00', dynamic=False) #monday
	pred0_ci = pred0.conf_int()
	table.append(pred0.predicted_mean[2:8 + 24*7])


# In[18]:


#predict count of passengers
	pred1 = results.get_prediction(start='2020-05-19 00:00:00', dynamic=False) #tuesday
	pred1_ci = pred1.conf_int()
# table.append(pred1.predicted_mean)
	table.append(pred1.predicted_mean[2:8 + 24*7])


# In[19]:


#predict count of passengers
	pred2 = results.get_prediction(start='2020-05-20 00:00:00', dynamic=False) #wednesday
	pred2_ci = pred1.conf_int()
# table.append(pred2.predicted_mean)
	table.append(pred2.predicted_mean[2:8 + 24*7])


# In[ ]:


#predict count of passengers
	pred3 = results.get_prediction(start='2020-05-21 00:00:00', dynamic=False) #thursday
	pred3_ci = pred1.conf_int()
# table.append(pred3.predicted_mean)
	table.append(pred3.predicted_mean[2:8 + 24*7])


# In[ ]:


#predict count of passengers
	pred4 = results.get_prediction(start='2020-05-22 00:00:00', dynamic=False) #friday
	pred4_ci = pred1.conf_int()
# table.append(pred4.predicted_mean)
	table.append(pred4.predicted_mean[2:8 + 24*7])


# In[ ]:


#predict count of passengers
	pred5 = results.get_prediction(start='2020-05-23 00:00:00', dynamic=False) #sat
	pred5_ci = pred1.conf_int()
# table.append(pred5.predicted_mean)
	table.append(pred5.predicted_mean[2:8 + 24*7])


# In[ ]:


#predict count of passengers
	pred6 = results.get_prediction(start='2020-05-24 00:00:00', dynamic=False) #sun
	pred6_ci = pred1.conf_int()
# table.append(pred6.predicted_mean)
	table.append(pred6.predicted_mean[2:8 + 24*7])


# In[25]:


	pred0.predicted_mean


# In[ ]:


	return('ok')


# In[ ]:




