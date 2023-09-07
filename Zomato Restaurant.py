#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df1=pd.read_excel(r"C:\Users\sahee\OneDrive\Desktop\Det_Saheel\Study\Project phase evaluation\documents\FP_E_P_3\Country-Code.xlsx")


# In[10]:


df2=pd.read_csv(r"C:\Users\sahee\OneDrive\Desktop\Det_Saheel\Study\Project phase evaluation\documents\FP_E_P_3\zomato.csv",encoding='latin-1')


# In[12]:


data= pd.merge (df1,df2)


# In[13]:


data.head()


# In[14]:


#countries in which zomato is present
data.Country.value_counts()


# In[17]:


#Store the country name
country_names=data.Country.value_counts().index
#Store the count value
country_count=data.Country.value_counts().values
#Plot a pie chart showing the countries that use Zomato
plt.pie(country_count, labels=country_names, autopct='%1.1f%%', shadow=True)


# In[20]:


#Plot a pie chart showing the top 3 countries that use Zomato
plt.pie(country_count[:3], labels=country_names[:3], autopct='%1.2f%%', shadow=True)

According to the pie-chart above, the top three countries that use Zomato are India, the United States, and the United Kingdom.
# In[21]:


#Grouping data 
data.groupby(['Aggregate rating', 'Rating color', 'Rating text']).size()


# In[22]:


data.groupby(['Aggregate rating', 'Rating color', 'Rating text']).size().reset_index()


# In[25]:


data.groupby(['Aggregate rating', 'Rating color', 'Rating text']).size().reset_index()


# In[26]:


#Rename the last count column from '0' to 'Rating Count' 
data_ratings=data.groupby(['Aggregate rating', 'Rating color', 'Rating text']).size().reset_index().rename(columns={0: 'Rating Count'})

data_ratings


# In[28]:


import matplotlib
matplotlib.rcParams['figure.figsize']=(12,8)
sns.barplot(x='Aggregate rating', y='Rating Count', data=data_ratings, hue='Rating color', palette=['Blue', 'Red', 'Orange', 'Yellow', 'Green', 'Green'])


# In[31]:


#Countries that did not give any rating
data[data['Rating color']=='White'].Country.value_counts()


# In[34]:


#Availability of online delivery
data_online_delivery=data[['Has Online delivery','Country']].groupby(['Has Online delivery','Country']).size()
print(data_online_delivery)


# In[36]:


sns.countplot(x="Rating color",data=data,palette=['blue','red','orange','yellow','green','green'])


# In[38]:


Excellent=data[data["Rating color"]=="Dark Green"].groupby("Country").size().sort_values(ascending=False).reset_index().rename(columns={0:"Rating_count"})
Excellent


# In[39]:


sns.barplot(x="Country", y="Rating_count" ,data=Excellent)


# In[40]:


poor=data[data["Rating color"]=="Red"].groupby("Country").size().sort_values(ascending=False).reset_index().rename(columns={0:"Rating_count"})
poor


# In[41]:


sns.barplot(x="Country", y="Rating_count" ,data=poor)

Find out which currency is used by which country?
# In[42]:


data.groupby(['Country','Currency']).size().sort_values(ascending=False).reset_index().rename(columns={0:"Rating_count"})


# In[43]:


# Which Countries do have online deliveries option?
data[data["Has Online delivery"]=="Yes"].groupby("Country").size().reset_index().rename(columns={0:"Rating_count"})


# In[44]:


cuisines=data.groupby(["Cuisines"]).size().sort_values(ascending=False).head(12).reset_index().rename(columns={0:"Rating_count"})
cuisines


# In[45]:


city=cuisines=data.groupby(["City"]).size().sort_values(ascending=False).head(10).reset_index().rename(columns={0:"Rating_count"})
city


# In[46]:


# top 10 restaurants
Rests=data.groupby(["Votes","Restaurant Name",]).size().reset_index().tail(10).rename(columns={0:"Rating_count"})
Rests


# In[47]:


sns.barplot(x="Restaurant Name",y="Votes",data=Rests)


# In[48]:


# To find the city with most rated restaurants
Top=data.groupby(["Votes","Restaurant Name","City"]).size().reset_index().tail(50).rename(columns={0:"Rating_count"})
Top


# In[49]:


top_final=Top.groupby(["City"]).size().sort_values(ascending=False).reset_index(name="Rating_count")
top_final


# In[56]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.graph_objects as px


# In[57]:


labels = list(data.Country.value_counts().index)
values = list(data.Country.value_counts().values)

fig = go.Figure(data = [go.Pie(labels=labels,values=values,rotation=90,hole = 0.5)], layout=go.Layout(title="Zomato's Worldwide Business",))
fig.show()


# In[60]:


data['Has Online delivery'].value_counts()


# In[61]:


data.drop(['Country Code', 'Address', 'Locality Verbose', 'Currency', 'Rating color', 'Is delivering now', 'Switch to order menu', 'Country'], axis=1, inplace=True)
data.head()


# In[67]:


data.describe()


# In[69]:


data.shape


# In[70]:


data.info()


# In[62]:


data.rename(columns={'Restaurant ID':'res_id','Restaurant Name':'res_name','City':'city','Locality':'locality','Longitude':'longitude','Latitude':'latitude','Cuisines':'cuisines','Average Cost for two':'price_for_two','Has Table booking':'table_prebooking','Has Online delivery':'delivers_online','Price range':'price_category','Aggregate rating':'avg_rating','Rating text':'rating_text','Votes':'votes'}, inplace=True)


# In[63]:


# Defining a function to calculate the number of cuisines offered by a particular restaurant. It will return an integer.

def number_of_cusines_per_restaurant(cuisines):
    return len(cuisines.split())


# In[64]:


data['num_cuisines_offered'] = data.cuisines.apply(number_of_cusines_per_restaurant)


# In[65]:


data


# In[66]:


data['city'].value_counts()


# In[71]:


plt.figure(figsize=(15,15))
sns.scatterplot(x='avg_rating', y='price_for_two', data=data, hue='table_prebooking');


# In[72]:


plt.figure(figsize=(15,15))
sns.scatterplot(x='avg_rating', y='price_for_two', data=data, hue='delivers_online');


# In[73]:


city_avg_df = data.groupby('city').mean().reset_index()
city_avg_df


# In[74]:


plt.figure(figsize=(15,20))
sns.barplot(y='city',x='price_for_two',data=city_avg_df.sort_values('price_for_two'))
plt.title('City vs Average Price for Two');


# In[80]:


fig = go.Figure()

fig.add_trace(go.Box(x=data.city, y=data.price_for_two, notched=True))
fig.update_layout(height=800, width=1400, title_text='Price for Two Distribution in Cities')


# In[81]:


plt.figure(figsize=(15,20))
sns.barplot(y='city',x='avg_rating',data=city_avg_df.sort_values('avg_rating'))
plt.title('City vs Average Rating');


# In[83]:


fig = go.Figure()

fig.add_trace(go.Box(x=data.city, y=data.avg_rating, notched=True))
fig.update_layout(height=800, width=1400, title_text='Rating Distribution in Cities')

Most of Zomato's business is concentrated in India, no other country even comes close.
Even within India, there is a huge bias. Most of Zomato's business is concentrated in and around New Delhi.
The popularity of the North Indian Cuisine also supports the above inference.
It can be noted that a majority of restaurants are priced at an average below Rs.2000. This proves that there are far too many people who spend an average of Rs.2000 or less for two.
It can also be noted that most restaurants that are priced below Rs.1000-Rs.1200 range, offer online order and delivery services, but no prebooking of tables.
And it starts becoming the opposite, if we start moving up the price range.
The optimum number of cuisines to be offered by restaurants is around 2-5 Cuisines.
Maximum number of people prefer going to restaurants offering 2-5 Cuisines, which also supports the above inference.
Although there are still a lot of people that visit restaurants offering more than 5 Cuisines.