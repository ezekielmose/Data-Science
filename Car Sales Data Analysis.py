#!/usr/bin/env python
# coding: utf-8

# Import all Libraries

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the Data

# In[60]:


cars_ds = pd.read_csv('E:\Data Science Projects\Datasets/cars_data.csv', encoding='latin1')


# In[71]:


cars_ds.head()


# In[76]:


cars_ds.size


# In[74]:


cars_ds.shape


# In[77]:


cars_ds.info()


# 1. We can observe that addressline2, state, postalcode, and territory have missing values
# 2. Moreover, all the column names are in upper case which needs to be converted into lower case. We will change data type of ORDERDATE

# In[78]:


cars_df.isnull().sum()


# In[79]:


cars_ds.describe()


# 1. The data describes that the maximum price is $100.00 and minimum price is $26.88. 
# 2. Additionally, the maximum MSRP is $214.00 and minimum MSRP is $33.00. 

# **Data Wrangling**

# In[87]:


column_name_lower=[]
for i in cars_ds.columns:
    column_name_lower.append(i.lower()) 
column_name_lower
for i in range(0, len(cars_ds.columns), 1):
    cars_ds=cars_ds.rename(columns={cars_ds.columns[i] : str(column_name_lower[i])})

cars_ds.head(3)


# *Explore Null Values and Duplicates*

# In[103]:


cars_ds.duplicated().sum()
print('There are ', cars_ds.duplicated().sum(), ' rows duplicate')


# In[104]:


isnull=pd.DataFrame(cars_ds.isnull().sum())
isnull.style.background_gradient(cmap='Blues')


# In[105]:


cars_ds.groupby(by=['country']).value_counts(['state'])


# In[106]:


print('Country has n rows data missing state:')
cars_ds[cars_ds['state'].isnull()]['country'].value_counts()


# *Investigate the territory null value*

# In[99]:


print('Below country has no territory label:')
cars_ds[cars_ds['territory'].isnull()]['country'].value_counts()


# *Look into current Territory Label*

# In[107]:


print('Current territory label:')
cars_ds['territory'].unique()


# *when analyze geographically we will look into 'country' and 'city' rather than territory and state because these two later columns have more missing values*

# *Change orderdate to datetime object*

# In[108]:


cars_ds['orderdate']=pd.to_datetime(cars_ds['orderdate'])
cars_ds['orderdate'].info()


# In[112]:


def plot_histogram(data, variable_name):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {variable_name}')
    plt.xlabel(variable_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

plot_histogram(cars_ds['sales'], 'Quantity')


# *Create new column weekday/month/year+quarter*

# In[109]:


cars_ds['monthvar']=cars_ds['orderdate'].dt.strftime('%b')
cars_ds['weekday']=cars_ds['orderdate'].dt.strftime('%a')

cars_ds.head(5)


# In[110]:


#Revenue by Year
yearly_sales=cars_ds.groupby(['year_id'])[['sales']].sum(numeric_only=True).reset_index()
yearly_sales


# In[111]:


custom_colors = ["#FF5733", "#FFC300", "#C70039", "#900C3F", "#581845"]

# Plot the yearly sales
plt.figure(figsize=(10, 6))
plt.title('Yearly Sales')

# Use the custom color palette
ax = sns.barplot(data=yearly_sales, x='year_id', y='sales', palette=custom_colors)

# Add labels to the bars
labels = [f'{value/1e6:.2f}M' for value in yearly_sales['sales']]
for i, label in enumerate(labels):
    ax.text(i, yearly_sales.loc[i, 'sales'], label, ha='center', fontsize=10)

plt.show()


# In[113]:


#Top Products by Country
sales_by_region = cars_ds.groupby(['country', 'productline'])['sales'].sum().reset_index()
top_products_by_region = sales_by_region.groupby('country').apply(lambda x: x.nlargest(3, 'sales'))
top_products_by_region


# In[114]:


#Overall country sales in each year
yearly_sales_by_country=cars_ds.groupby(['country', 'year_id'])[['sales']].sum().reset_index()

yearly_sales_by_country.head(20)


# In[115]:


# Plot the revenue by year and country
plt.figure(figsize=(12, 4))
plt.title('Yearly Sales by Country')
ax = sns.barplot(data=yearly_sales_by_country, x='country', y='sales', hue='year_id', palette=custom_colors)
ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
plt.show()


# 2004 has the highest revenue and USA is the top contribution followed by France and Spain.

# In[122]:


#Top 10 Sales Country by each year
_2003sales_by_country_ds=cars_ds[cars_ds['year_id']==2003].groupby(['country', 'year_id'])[['sales']].sum().sort_values('sales',ascending= False).reset_index().head(10)
_2004sales_by_country_ds=cars_ds[cars_ds['year_id']==2004].groupby(['country', 'year_id'])[['sales']].sum().sort_values('sales',ascending= False).reset_index().head(10)
_2005sales_by_country_ds=cars_ds[cars_ds['year_id']==2005].groupby(['country', 'year_id'])[['sales']].sum().sort_values('sales',ascending= False).reset_index().head(10)


# In[123]:


#Took the table aggregate earlier to calculate Revenue percenatge
yearly_sales=cars_ds.groupby(['year_id'])[['sales']].sum(numeric_only=True).reset_index()
yearly_sales


# In[124]:


#Arrange the yearly top 10 and it's percentage in a table to see their change

_2003sales_by_country_ds['percent1']= round((_2003sales_by_country_ds['sales']/ (yearly_sales['sales'][0])*100), 2)
_2004sales_by_country_ds['percent2']= round((_2004sales_by_country_ds['sales']/ (yearly_sales['sales'][1])*100), 2)
_2005sales_by_country_ds['percent3']= round((_2005sales_by_country_ds['sales']/ (yearly_sales['sales'][2])*100), 2)

Top10_2003_2005=pd.concat([_2003sales_by_country_ds,_2004sales_by_country_ds,_2005sales_by_country_ds], axis=1)
Top10_2003_2005


# In[125]:


#See how much does top 10 contribute to the total revenue 
print( '2003 top 10 countries takes up to', round(Top10_2003_2005['percent1'].sum(), 2), '% of the revenue')
print( '2004 top 10 countries takes up to', round(Top10_2003_2005['percent2'].sum(), 2), '% of the revenue')
print( '2005 top 10 countries takes up to', round(Top10_2003_2005['percent3'].sum(), 2), '% of the revenue')


# 1. We observe that USA is an important market for the company and Norway was top 4 in 2003, but not in top 10 in the next 2 years, however Japan quickly came up to top 10 after 2004, it might related to the company market decision

# **Monthly and Weekly Revenue Trend**

# In[128]:


#Revenue by month
monthly_revenue = df.groupby(['month_id', 'year_id'])[['sales']].sum().reset_index()
ax = sns.barplot(data=monthly_revenue, x='month_id', y='sales', hue='year_id', palette=custom_colors)
plt.title('Monthly Revenue')
ax.set_xlabel('Month')
plt.show()


# In[129]:


#Revenue by week
order=['Mon','Tue','Wed', 'Thu','Fri','Sat', 'Sun']
weekly_revenue=df.groupby(['weekday', 'year_id'])[['sales']].sum().reset_index()
ax=sns.barplot(data=weekly_revenue, x='weekday', y='sales', hue='year_id', palette=custom_colors)
plt.title('weekly_revenue')
ax.set_xlabel('weekday')
ax.set_xticklabels(order)
weekly_revenue


# 1. There are only 5 months data in 2005, so it's not intact, we should consider while looking at yearly revenue
# 2. From monthly perspective, the second half year (Jul-Dec) has a speedy growth in sales then the first half it might be a purchasing season for this industry
# 3. From a weekly perspective, Thursday has the lowest buy rate thorughout a week, while closer to weekend the stronger purchasing power is than weekday

# **EDA - Analysis of Product line and Shipping Status**

# In[130]:


cars_ds.columns


# In[131]:


#Make a plot function 

def barplotter(data, colname1, colname2, title, **kwargs):
    plt.title(title)
    sns.barplot(data=data, x=data[colname1], y=data[colname2], palette='Paired', **kwargs)
    plt.xticks(rotation=60)


# In[132]:


# Plot each product line sales
sales_by_productline = df.groupby(['productline'])[['sales']].sum().reset_index()
barplotter(data=sales_by_productline, colname1='productline', colname2='sales', title='2003-2005 Revenue by Productline')
plt.show()


# In[133]:


# Ordered quantity by each product line
quantityordered_by_productline = df.groupby(['productline'])[['quantityordered']].sum().reset_index()
barplotter(data=quantityordered_by_productline, colname1='productline', colname2='quantityordered', title="2003-2005 Q'ty by Productline")
plt.show()


# In[134]:


#Revenue and Q'ty ordered have similar trend, classic car is the most popular, Vintage car is the second"/"
"I'd like to look at unit price and q'ty distribution, sometimes the higher unit price the lower q'ty it is"


# In[135]:


# Unit Price Yearly
sns.scatterplot(data=df, x='priceeach', y='quantityordered', hue='year_id')
plt.title("Unit Price x Q'ty Yearly Distribution")
plt.xlabel('unirprice')
plt.ylabel("Q'ty")


# In[136]:


#Plot the unitprice and productline

plt.title("Unit Price Among Different Products")
plt.xlabel('Unirprice')
plt.ylabel("Q'ty")
sns.histplot(data=df, x='priceeach', hue='productline', multiple='fill', bins=range(26, 101, 5), palette='Set2')


# In[137]:


sns.scatterplot(data=df, x='priceeach', y='msrp', hue='dealsize')
sns.lineplot(x=(26, 26), y=(100, 100), linestyle='--', color='r')
plt.title("Unit Price x msrp Distribution")
plt.xlabel('unit price')
plt.ylabel("msrp")


# 1. It's interetsing to see from "Unit Price x Q'ty Yearly Distribution" that 2003 and 2004 order q'ty are pretty stable around 20-50
# 2. While 2005 some order q'ty jump out of 20-50pcs, a little bit less and more, which we cannot see in the last 2 years
# 3. From plot"Unit Price Among Different Products", Vintage car usually sells at lower price ($30-$50), plane and ships usually sells at higher price($60-$90), and Train $45-70,
# 4. Motorcycle and classic cars have relative same proportion at each price range"""

# In[138]:


#Product line by country
countrysales_by_productline=df.groupby(['country', 'dealsize'])[['sales']].sum().reset_index()
countrysales_by_productline
#Plot Countrysales by productline
plt.figure(figsize=(20, 5))
barplotter(data=countrysales_by_productline, colname1='country', colname2='sales', title='Deal Size by Country', hue=countrysales_by_productline['dealsize'])


# 1. Most countries have medium deal size
# 2. Small deal size is more than large deal size

# **Order Status Analysis**

# In[139]:


df['status'].unique()


# In[140]:


# Extract status that are not shipped
df_other_status= df[~(df['status']=='Shipped')]

#Plot them in yearly manner
sns.histplot(data=df_other_status, x='status', hue='year_id', multiple='stack')


# 1.Lastly I take a look at the order status, it can help to check if any orders are left behind
# 2.Overall it looks good, mostly the orders that need to be handled (i.e In Proccess, Disputed) are in this year (assuming data collecting year is 2005)
# 3.Only one order from 2004 is on hold, we can ask stakeholder to look after it

# Conclusions¶
# 1. 2005 Sales data only have 5 months, it need to be considered while checking yearly revenue
# 2. Top 10 countries supply over 80-95% revenue
# 3. The most popular product line is classic cars, and the biggest market is USA
# 4. In 2003&2004 most order q'ty are around 20-50pcs, in 2005 we can see some orders q'ty are more than that section
# 5. Deal size distribution : Medium > Small > Large
# 6. Vintage car usually sells at lower price ($30-$50), Train $45-70, Plane and ships usually sells at higher price($60-$90)
# 7. Motorcycle and classic cars have relative same proportion at each price range 7.Second-half year (Jul-Dec) has a speedy growth in sales then the first half, it might be a purchasing season for this industry
# 8. From a weekly perspective, Thursday has the lowest buy rate thorughout a week, while closer to weekend stronger the purchasing power is than weekday
# 9. There a few msrp(manufactured suggest resell price) lower than unit price, usually it stands for distrbutor is doing a money-losing business, it deserves further investigation
# 10. Among orders that are not shipped, one 2004 order is on hold
# 11. There are some data missing such as addressline2, state, postalcode, territory
# 12. Analysis suggests add USA or North America as a territory label, also integrate Japan in APAC label
