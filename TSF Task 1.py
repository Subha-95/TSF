#!/usr/bin/env python
# coding: utf-8

# # <U>Data Science & Business AnalyticsTasks</u>
# <font color='BLUE'><b><h3>GRIP @ The Spart Foundation</h3></b></font>
# <font color='GREEN'><b>TASK 1 - Predict the percentage of an student based on the no. of study hours.</b></font>

# <font color='GREEN'><b><h2>Name - Subhashree Saha</h2></b></font>
# 
# Dataset Link : http://bit.ly/w-data
# 
# Github Link: https://github.com/Subha-95/TSF.git
# 
# Problem statement : <u>What will be predicted score if a student studies for 9.25 hrs/ day?</u>
# 
# 

# <h3><u><b>IMPORTING LIBRARIES</b></u></h3>

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <h3><u><b>IMPORTING DATASET</b></u></h3>

# Data set link: https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv

# In[5]:


train=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
train.shape


# In[6]:


train.head(5)


# In[7]:


train.describe()


# In[7]:


train.info()


# <h3><u><b>DISTRIBUTION OF SCORES IN GIVEN DATASET</b></u></h3>

# In[8]:


train.plot(x="Hours", y="Scores",style="o")
plt.title("Distribution of scores")
plt.xlabel("Study Hours") 
plt.ylabel("Obtained Score")
plt.show()


# <font color='GREEN'><b>By observing the graph we can say there is a strong linear relationship between scores and study hours as it resmbles a straight line. So this dataset is ideal to perform linear regression.</b></font>

# <h3><u><b>TRAIN AND TEST DATA SPLIT</b></u></h3>

# In[9]:


x1 = train.iloc[:,0].values
y1 = train.iloc[:,1].values
x = x1.reshape(-1,1)
y = y1.reshape(-1,1)


# <h3><u><b>RANDOM LINEAR REGRESSION MODEL</b></u></h3>
# <B>Applying on TEST data</B>

# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state=0)


# In[11]:


from sklearn.linear_model import LinearRegression
linearRegressor= LinearRegression()
linearRegressor.fit(x_train, y_train)


# In[12]:


line = linearRegressor.coef_*x+linearRegressor.intercept_
plt.scatter(x, y)
plt.plot(x, line, color="pink")
plt.show()


# <h3><u><b>ACCURACY SCORE FROM TRAINING AND TEST DATA</b></u></h3>

# In[13]:


print('Test Score')
print(linearRegressor.score(x_test, y_test))
print('Training Score')
print(linearRegressor.score(x_train, y_train))


# In[14]:


print(x_test)
y_pred = linearRegressor.predict(x_test)


# In[15]:


df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1


# <b>PREDICTION ON TEST AND TRAIN DATA</b>

# In[16]:


y_pred= linearRegressor.predict(x_test)
x_pred= linearRegressor.predict(x_train)


# <h3><u><b>SOLLUTION OF PROBLEM STATEMENT</b></u></h3>

# In[17]:


print('Score of student who studied for 9.25 hours/day', linearRegressor.predict([[9.25]]))


# <h3><u><b>SOLLUTION OF PROBLEM STATEMENT</b></u></h3>

# In[19]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))


# <blockquote><h4><b><font color="green">AFTER BUILDING THE LINEAR REGRESSION MODEL AND OBTAIN THE PREDICTION WE CAN CONCLUDE "PREDICTED SCORE OF A STUDENT STUDIES WHO STUDY 9.25 HRS/ DAY WILL BE ABLE TO SCORE 92.91505723"</font></b></h4> </blockquote>
