#!/usr/bin/env python
# coding: utf-8

# # World Happiness Report : 2021 Analysis

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


sns.set_style('darkgrid')
plt.rcParams['font.size']=15
plt.rcParams['figure.figsize']=(10,7)
plt.rcParams['figure.facecolor']='FFE5B4' # hexadecimal code for peach color


# ### Data Acquisition

# In[3]:


data=pd.read_csv('world-happiness-report-2021.csv')


# In[4]:


data.shape


# In[5]:


# Display the top 5 rows of the Data
data.head()


# ### Data Preprocessing

# In[6]:


data.rename(columns={'Ladder score':'Happiness score','Standard error of ladder score':'Std. error of Happiness score'},inplace=True)


# In[7]:


data


# In[8]:


data_columns = ['Country name','Regional indicator','Happiness score','Logged GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption']


# In[9]:


data=data[data_columns].copy()


# In[10]:


happy_df=data.rename({'Country name':'country_name','Regional indicator':'region','Happiness score':'happiness_score','Logged GDP per capita':'logged_GDP_per_capita','Social support':'social_support','Healthy life expectancy':'health_life_expectancy','Freedom to make life choices':'freedom_to_make_lifechoices','Generosity':'generosity','Perceptions of corruption':'perceptions_of_corruption'},axis=1)


# In[11]:


# Displays top 5 happiness countries
happy_df.head()


# In[12]:


#Displays least 5 happiness countries
happy_df.tail()


# In[13]:


happy_df.drop_duplicates()


# In[14]:


display(happy_df.drop_duplicates())


# In[15]:


happy_df.info()


# In[16]:


happy_df['country_name'].unique()


# In[92]:


happy_df['region'].unique()


# ### Checking for Null Values

# In[14]:


happy_df.isnull().sum()


# In[43]:


mean = happy_df.mean()
print(mean)


# In[59]:


happy_df.describe()


# ### Happiness score distribution by country

# In[81]:


plt.figure(figsize=(10, 6))
plt.hist(happy_df['happiness_score'], bins=20, color='green', edgecolor='black')

plt.xlabel('Happiness Score')
plt.ylabel('Frequency')
plt.title('Happiness Score Distribution by Country')
plt.tight_layout()
plt.show()


# ### Regional Analysis

# #### No:of countries in each region

# In[56]:


total_country=happy_df.groupby('region')[['country_name']].count()
print(total_country)


# In[83]:


grouped_data = happy_df.groupby('region')
mean_happiness_score_by_region = grouped_data['happiness_score'].mean()
plt.figure(figsize=(10, 6))
mean_happiness_score_by_region.plot(kind='bar', color='m', edgecolor='black')

plt.xlabel('Region')
plt.ylabel('Mean Happiness Score')
plt.title('Mean Happiness Score by Region')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# #### Life expectancy of all countries in Sub-Saharan Africa region

# In[84]:


africa_data = happy_df[happy_df['region'] == 'Sub-Saharan Africa']
life_expectancy = africa_data['health_life_expectancy']
country_names = africa_data['country_name']
plt.figure(figsize=(10, 6))
plt.stackplot(country_names, life_expectancy, labels=country_names, color='orange',alpha=0.8)

plt.xlabel('Country')
plt.ylabel('Life Expectancy')
plt.title('Life Expectancy in Sub-Saharan Africa (Stacked Plot)')
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# #### Happiness score of all countries in South Asia region

# In[72]:


asia_data = happy_df[happy_df['region'] == 'South Asia']
plt.figure(figsize=(10, 6))
plt.plot(asia_data['country_name'],asia_data['happiness_score'], marker='o', color='red')

plt.xlabel('Country')
plt.ylabel('Happiness Score')
plt.title('Happiness Scores in South Asia')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()


# #### Percentage of Social support in all countries in Westren Europe Region

# In[77]:


europe_data = happy_df[happy_df['region'] == 'Western Europe']
mean_social_support = europe_data.groupby('country_name')['social_support'].mean()

plt.figure(figsize=(10, 7))
plt.pie(mean_social_support, labels=mean_social_support.index, autopct='%1.1f%%',)
 
plt.title('Distribution of Social Support in Western Europe (Mean)')
plt.show()


# #### Freedom to Make life choices of all countries in Middle East and North Africa region

# In[79]:


mena_data = happy_df[happy_df['region'] == 'Middle East and North Africa']
plt.scatter(mena_data['country_name'], mena_data['freedom_to_make_lifechoices'], color='black', marker='o')
plt.xlabel('Country')
plt.ylabel('Freedom to Make Life Choices')
plt.title('Freedom to Make Life Choices in the Middle East and North Africa')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ### Analysing factors affecting Happiness  in some of the countries

# #### 1.Finland

# In[86]:


data_finland = happy_df[happy_df['country_name'] == 'Finland']
mean_values1 = data_finland.mean()
mean_values1.plot(kind='bar', figsize=(10, 6), color='red', edgecolor='black')

plt.xlabel('Columns')
plt.ylabel('Mean Value')
plt.title('Mean Values of Columns for Finland')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# #### 2.India

# In[85]:


india_data = happy_df[happy_df['country_name'] == 'India']
mean_values2 = india_data.mean()
mean_values2.plot(kind='bar', figsize=(10, 6), color='pink', edgecolor='black')

plt.xlabel('Columns')
plt.ylabel('Mean Value')
plt.title('Mean Values of Columns for India')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# #### 3.Afganisthan

# In[87]:


af_data = happy_df[happy_df['country_name'] == 'Afghanistan']
mean_values3 = af_data.mean()
mean_values3.plot(kind='bar', figsize=(10, 6), color='blue', edgecolor='black')

plt.xlabel('Columns')
plt.ylabel('Mean Value')
plt.title('Mean Values of Columns for Afganisthan')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# ### Scatter plot between Happiness Score & GDP per capita based on region

# In[18]:


plt.rcParams['figure.figsize']=(15,7)
plt.title('Plot between Happiness score and GDP')
sns.scatterplot(x=happy_df.happiness_score,y=happy_df.logged_GDP_per_capita,hue=happy_df.region,s=200);

plt.legend(loc='upper left',fontsize='10')
plt.xlabel('Happiness Score')
plt.ylabel('GDP per capita')


# * In the above Scatter plot we can observe that the Westren europe region have most happiness score and highest GDP per capita and represented by blue dots
# * The Sub-saharan Africa region has least no:of countries

# ### GDP per capita by region

# In[19]:


gdp_region=happy_df.groupby('region')['logged_GDP_per_capita'].sum()
gdp_region


# ### Pie plot for plotting GDP distribution by Region

# In[20]:


gdp_region.plot.pie(autopct = '%1.1f%%')
plt.title('GDP by Region')
plt.ylabel('')


# * By the above Pie plot we can say that Sub-saharan Africa region contributes highest GDP beacuse the region consists of highest no:of countries in its region
# * North America and ANZ region contributes less percentage of GDP in the pie plot it consists of only 4 countries in the region 

# ### Correlation Map

# In[21]:


cor=happy_df.corr(method="pearson")
f, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(cor, mask=np.zeros_like(cor, dtype=bool),
            cmap="Blues", square=True, ax=ax)
plt.show()


# * The dark blue colors in the correlation map above indicates that those factors have highest correaltion or falls under positive correlation
# * The grey and white boxes indicates that those factors have low correlation or falls under negative correlation

# ### Corruption in each region

# In[22]:


corruption=happy_df.groupby('region')[['perceptions_of_corruption']].mean()
corruption


# * Central and Eastren Europe region has highest rate of perceptions of corruption
# * North america & ANZ and Western Europe region lowest rate of perceptions of corruption

# ### Bar Plot for plotting Corruption index in Each region

# In[23]:


plt.rcParams['figure.figsize']=(12,8)
plt.title('Perception of Corruption in various Regions')
plt.xlabel('Regions',fontsize=15)
plt.ylabel('Corruption Index',fontsize=15)
plt.xticks(rotation=30,ha='right')
plt.bar(corruption.index,corruption.perceptions_of_corruption)


# ### Bar plot for Life expectancy in Top 10 and Bottom 10 Countries

# In[24]:


top_10=happy_df.head(10)
bottom_10=happy_df.tail(10)


# In[25]:


fig,axes=plt.subplots(1,2,figsize=(16,6))
plt.tight_layout(pad=2)
xlabels=top_10.country_name
axes[0].set_title('Top 10 happiest countries life expectancy')
axes[0].set_xticklabels(xlabels,rotation=45,ha='right')
sns.barplot(x=top_10.country_name,y=top_10.health_life_expectancy,ax=axes[0])
axes[0].set_xlabel('Country name')
axes[0].set_ylabel('Life Expectancy')

xlabels=bottom_10.country_name
axes[1].set_title('Bottom 10 happiest countries life expectancy')
axes[1].set_xticklabels(xlabels,rotation=45,ha='right')
sns.barplot(x=bottom_10.country_name,y=bottom_10.health_life_expectancy,ax=axes[1])
axes[1].set_xlabel('Country name')
axes[1].set_ylabel('Life Expectancy')


# ### Happiness score b/w Freedom to make life choices

# In[26]:


plt.rcParams['figure.figsize']=(15,7)
sns.scatterplot(x=happy_df.freedom_to_make_lifechoices,y=happy_df.happiness_score,hue=happy_df.region,s=200)
plt.legend(loc='upper left',fontsize='12')
plt.xlabel('Freedom to make life choices')
plt.ylabel('Happiness Score')


# ### Analysing Perceptions of corruption in some of the Countries

# #### Analysing the top 10 countries with lowest corruption index

# In[27]:


country=happy_df.sort_values(by='perceptions_of_corruption').head(10)
plt.rcParams['figure.figsize']=(12,6)
plt.title('Countries with Most Perception of Corruption')
plt.xlabel('Country',fontsize=13)
plt.ylabel('Corruption Index',fontsize=13)
plt.xticks(rotation=30,ha='right')
plt.bar(country.country_name,country.perceptions_of_corruption)


# #### Analysing the top 10 countries with highest corruption of Index

# In[28]:


country=happy_df.sort_values(by='perceptions_of_corruption').tail(10)
plt.rcParams['figure.figsize']=(12,6)
plt.title('Countries with Most Perception of Corruption')
plt.xlabel('Country',fontsize=13)
plt.ylabel('Corruption Index',fontsize=13)
plt.xticks(rotation=30,ha='right')
plt.bar(country.country_name,country.perceptions_of_corruption)


# ### Scatter plot between Happiness score and Corruption

# In[29]:


plt.rcParams['figure.figsize']=(15,7)
sns.scatterplot(x=happy_df.happiness_score,y=happy_df.perceptions_of_corruption,hue=happy_df.region,s=200)
plt.legend(loc='lower left',fontsize='14')
plt.xlabel('Happiness Score')
plt.ylabel('Corruption')


# ### Pair plot for Happiness score and factors affecting the happiness score

# In[38]:


figsize=(15,10)
sns.pairplot(happy_df,hue='happiness_score')

