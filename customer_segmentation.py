import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'segmentation data.csv'
data = pd.read_csv(file_path)

data_cleaned = data.dropna().drop_duplicates()

labels = ['Male', 'Female']
data_sex = data_cleaned['Sex'].value_counts()

print(data_sex)

plt.figure(figsize=(13,10))
plt.title("Customers Sex")
plt.pie(x=data_sex, labels=labels, autopct='%.2f%%')
plt.show()

labels = ['single', 'non-single']
data_marital_status = data_cleaned['Marital status'].value_counts()

print(data_marital_status)
plt.figure(figsize=(13,10))
plt.title('Customers Marital Status')
plt.pie(x = data_marital_status, labels=labels, autopct='%.2f%%')
plt.show()

data_age = data_cleaned['Age'].value_counts().sort_index()
labels = data_cleaned['Age'].value_counts().sort_index().index

plt.figure(figsize=(15,10))
plt.title('Customers Ages')
plt.bar(x=labels, height=data_age)
plt.show()

data_education = data_cleaned['Education'].value_counts().sort_index()
labels = ['other / unknown', 'high school', 'university', 'graduate school']

plt.figure(figsize = (13, 10))
plt.title("Customer's Education")
plt.pie(x = data_education, labels=labels, autopct='%.2f%%')
plt.show()

data_income = data_cleaned['Income']
print('minimum income customers: {}\nmaximum income customers: {}\naverage income customers: {}'.format(min(data_income), max(data_income), np.average(data_income)))
data_occupation = data_cleaned['Occupation'].value_counts().sort_index()
labels = ['unemployed / unskilled', 'skilled employed / official', 'management / self-employed / highly qualified employee / officer']

plt.figure(figsize=(13,10))
plt.title("customer's Occupatiopn")
plt.pie(x= data_occupation, labels=labels, autopct='%.2f%%')
plt.show()

data_settlement_size =  data_cleaned['Settlement size'].value_counts().sort_index()
labels = ['small city', 'mid-sized city', 'big city']

plt.figure(figsize=(13,10))
plt.title("Customer's Settlement Size")
plt.pie(x = data_settlement_size, labels=labels,autopct ='%.2f%%')
plt.show()
