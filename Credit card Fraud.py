#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import ks_2samp
from tqdm import tqdm


# In[2]:


#importing credit fraud CSV file and recalling it
df= pd.read_csv("C:\\Users\\Ammad\OneDrive\\Desktop\\BPP University\\other courses\\Machine Learning 1\\creditcard.csv")
df


# In[3]:



#class colums the status of the transaction weather it was a fraud or it was a real transaction.
#we will target the class
target= "Class"
df[target].value_counts()


# In[ ]:





# In[ ]:





# In[4]:


#installing snap lib for data cleaning 
get_ipython().system('pip -q install snaplib')
get_ipython().system('pip install termcolor')
get_ipython().system('pip install lightgbm')


# In[5]:


#loading snaplib libraray
from snaplib.snaplib import Snaplib


# In[6]:


#data cleaning to eliminate irrelevant values
#sl needs to be insatlled to clean data 
df=Snaplib.cleane(df,target=target,verbose= True)
df[target].value_counts


# In[7]:


#cleaning data from the table by removing duplicates and removing not a number values
df.info()
df = Snaplib.cleane(df, target='Class', verbose=True)


# In[8]:


# Set the aesthetic style of the plots
sns.set_style("darkgrid")
#plotting graph
f,(axis1,axis2)=plt.subplots(2,1, sharex= True, figsize=(11,5))
bins = 70

# Plotting for Fraud transactions
axis1.hist(df.Time[df.Class == 1], bins=bins, color='red', alpha=0.7)
axis1.set_title('Fraud', fontsize=14)
axis1.set_ylabel('Transaction Count', fontsize=12)
axis1.grid(True, linestyle='-', alpha=0.5)

# Plotting for Normal transactions
axis2.hist(df.Time[df.Class == 0], bins=bins, color='green', alpha=0.7)
axis2.set_title('Normal', fontsize=14)
axis2.set_xlabel('Time (in Seconds)', fontsize=12)
axis2.set_ylabel('Transaction Count', fontsize=12)
axis2.grid(True, linestyle='-', alpha=0.5)

# Remove top and right spines
for axis in [axis1, axis2]:
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


# In[10]:


#to set fields in respective to create a histogram grid in future
target = "Class"

predictors = df.columns.to_list()
predictors.remove(target)
print(predictors)


# In[12]:


#creating a grid of histograms using seaborn for multiple predictor variables (specified by the predictors list) with respect to the target variable. 
fig, axis = plt.subplots(nrows=6, ncols=5, figsize=(18,18))
axis = axis.flatten()

for idx, axis in tqdm(enumerate(axis)):
    try:
        sns.histplot(data=df, x=df[predictors].iloc[:, idx],
                     ax=axis, hue=target, legend=True, bins=20)
#         sns.lmplot(data=df, x='Amount', y=predictors[idx], hue=target)
        axis.set_ylabel('')    
        axis.set_xlabel('')
        axis.set_title(predictors[idx])
    except(IndexError):
        pass


# In[ ]:

/




