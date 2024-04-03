#!/usr/bin/env python
# coding: utf-8

# ### Food Delivery Costs and Profitability Analysis
# 

# In[1]:


import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


# In[2]:


food_ds=pd.read_csv("F:\Data Science Personal Projects/food_orders_new_delhi.csv")
food_ds.head()


# In[3]:


food_ds.info ()


# In[4]:


food_ds.isnull().sum()


# In[5]:


food_ds.describe ()


# In[6]:


from datetime import datetime

# convert date and time columns to datetime
food_ds['Order Date and Time'] = pd.to_datetime(food_ds['Order Date and Time'])
food_ds['Delivery Date and Time'] = pd.to_datetime(food_ds['Delivery Date and Time'])
# first, let's create a function to extract numeric values from the 'Discounts and Offers' string
def extract_discount(discount_str):
    if 'off' in discount_str:
        # Fixed amount off
        return float(discount_str.split(' ')[0])
    elif '%%' in discount_str:
        # Percentage off
        return float(discount_str.split('%%')[0])
    else:
        # No discount
        return 0.0
    # apply the function to create a new 'Discount Value' column
food_ds['Discount Percentage'] = food_ds['Discounts and Offers'].apply(lambda x: extract_discount(x))
# for percentage discounts, calculate the discount amount based on the order value
food_ds['Discount Amount'] = food_ds.apply(lambda x: (x['Order Value'] * x['Discount Percentage'] / 100)
                                                   if x['Discount Percentage'] > 1
                                                   else x['Discount Percentage'], axis=1)
# adjust 'Discount Amount' for fixed discounts directly specified in the 'Discounts and Offers' column
food_ds['Discount Amount'] = food_ds.apply(lambda x: x['Discount Amount'] if x['Discount Percentage'] <= 1
                                                   else x['Order Value'] * x['Discount Percentage'] / 100, axis=1)
food_ds[['Order Value', 'Discounts and Offers', 'Discount Percentage', 'Discount Amount']].head(), food_ds.dtypes


# In[7]:


food_ds.head()


# In[8]:


# calculate total costs and revenue per order
food_ds['Total Costs'] = food_ds['Delivery Fee'] + food_ds['Payment Processing Fee'] + food_ds['Discount Amount']
food_ds['Revenue'] = food_ds['Commission Fee']
food_ds['Profit'] = food_ds['Revenue'] - food_ds['Total Costs']

#aggregating data to get overall metrics
total_orders = food_ds.shape[0]
total_revenue = food_ds['Revenue'].sum()
total_costs = food_ds['Total Costs'].sum()
total_profit = food_ds['Profit'].sum()

overall_metrics = {
    "Total Orders": total_orders,
    "Total Revenue": total_revenue,
    "Total Costs": total_costs,
    "Total Profit": total_profit
}

overall_metrics


# In[9]:


# pie chart for the proportion of total costs
costs_breakdown = food_ds[['Delivery Fee', 'Payment Processing Fee', 'Discount Amount']].sum()
plt.figure(figsize=(5, 9))
plt.pie(costs_breakdown, labels=costs_breakdown.index, autopct='%1.1f%%', startangle=90, colors=['green', 'Orange', 'blue'])
plt.title('Proportion of Total Costs for Food Delivery')
plt.show()


# In[10]:


# bar chart for total revenue, costs, and profit
totals = ['Total Revenue', 'Total Costs', 'Total Profit']
values = [total_revenue, total_costs, total_profit]

plt.figure(figsize=(8, 6))
plt.bar(totals, values, color=['b', 'g', 'r'])
plt.title('Bar Chart for Total Revenue, Costs, and Profit')
plt.ylabel('Amount (INR)')
plt.show()


# In[13]:


# histogram of profits per order
plt.figure(figsize=(10, 6))
plt.hist(food_ds['Profit'], bins=50, color='orange', edgecolor='black')
plt.title('Profit Distribution per Order in Food Delivery')
plt.xlabel('Profit')
plt.ylabel('Number of Orders')
plt.axvline(food_ds['Profit'].mean(), color='red', linestyle='dashed', linewidth=1)
plt.show()


# In[20]:


import warnings
warnings.simplefilter("ignore")


# In[21]:


# filter the dataset for profitable orders
profitable_orders = food_ds[food_ds['Profit'] > 0]

# calculate the average commission percentage for profitable orders
profitable_orders['Commission Percentage'] = (profitable_orders['Commission Fee'] / profitable_orders['Order Value']) * 100

# calculate the average discount percentage for profitable orders
profitable_orders['Effective Discount Percentage'] = (profitable_orders['Discount Amount'] / profitable_orders['Order Value']) * 100

# calculate the new averages
new_avg_commission_percentage = profitable_orders['Commission Percentage'].mean()
new_avg_discount_percentage = profitable_orders['Effective Discount Percentage'].mean()

new_avg_commission_percentage, new_avg_discount_percentage


# In[22]:


# simulate profitability with recommended discounts and commissions
recommended_commission_percentage = 22.0 # 30%
recommended_discount_percentage = 5.0    # 6%

# calculate the simulated commission fee and discount amount using recommended percentages
food_ds['Simulated Commission Fee'] = food_ds['Order Value'] * (recommended_commission_percentage / 100)
food_ds['Simulated Discount Amount'] = food_ds['Order Value'] * (recommended_discount_percentage / 100)

# recalculate total costs and profit with simulated values
food_ds['Simulated Total Costs'] = (food_ds['Delivery Fee'] +
                                        food_ds['Payment Processing Fee'] +
                                        food_ds['Simulated Discount Amount'])

food_ds['Simulated Profit'] = (food_ds['Simulated Commission Fee'] -
                                   food_ds['Simulated Total Costs'])

# visualizing the comparison
import seaborn as sns

plt.figure(figsize=(14, 7))

# actual profitability
sns.kdeplot(food_ds['Profit'], label='Actual Profitability', fill=True, alpha=0.5, linewidth=2)

# simulated profitability
sns.kdeplot(food_ds['Simulated Profit'], label='Estimated Profitability with Recommended Rates', fill=True, alpha=0.5, linewidth=2)

plt.title('Comparison of Profitability in Food Delivery: Actual vs. Recommended Discounts and Commissions')
plt.xlabel('Profit')
plt.ylabel('Density')
plt.legend(loc='upper left')
plt.show()


# In[ ]:




