#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def read_file(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data = pd.read_excel(file_path)
    elif file_path.endswith('.sql'):
        if db_connection is None:
            raise ValueError("Please provide a database connection string.")
        engine = create_engine(db_connection)
        query = "SELECT * FROM your_table_name"
        data = pd.read_sql(query, engine)    
    else:
        raise ValueError('File type not supported. Please provide a CSV or Excel file.')
    return data


# In[3]:


def preprocess_data(data):
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = identify_categorical_features(data)  # Assuming you have this function defined

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    data[numerical_features] = imputer.fit_transform(data[numerical_features])

    # One-hot encode categorical features
    data = pd.get_dummies(data, columns=[feature for feature, _ in categorical_features])

    # Scale numerical features
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data

def identify_categorical_features(data):
    categorical_features = data.select_dtypes(include=['object']).columns
    binary_features = []

    for feature in categorical_features:
        unique_values = data[feature].nunique()
        if unique_values == 2:
            binary_features.append((feature, 'binary'))
        else:
            binary_features.append((feature, 'categorical'))

    return binary_features


# In[47]:


from matplotlib.backends.backend_pdf import PdfPages
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime as dt
import datetime as dt
import numpy as np
# Define your data loading and preprocessing functions here

def create_scatter_plot(data, feature):
    fig = px.scatter(data, x=feature, y='Total', title=f'Scatter Plot of {feature}')
    
    return fig

def create_pie_plot(data, feature):
    feature_counts = data[feature].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(feature_counts, labels=feature_counts.index.values.tolist(), autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(f'Pie Plot of {feature}')
    plt.show()

def create_bar_plot(data, feature):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=feature)
    plt.title(f'Bar Plot of {feature}')
    plt.xticks(rotation=45)
    plt.show()

def create_histogram(ax, data, feature):
    sns.histplot(data[feature], bins=20, kde=True, ax=ax)
    ax.set_title(f'Histogram of {feature}')
    plt.tight_layout()
    plt.show()

def create_box_plot(ax, data, feature):
    sns.boxplot(x=data[feature], ax=ax)
    ax.set_title(f'Box Plot of {feature}')
    plt.tight_layout()
    plt.show()

def create_visualizations_for_column(data, column_name):
    fig_histogram, ax_histogram = plt.subplots(figsize=(8, 6))
    create_histogram(ax_histogram, data, column_name)

    fig_box_plot, ax_box_plot = plt.subplots(figsize=(8, 6))
    create_box_plot(ax_box_plot, data, column_name)

    fig_scatter_plot = create_scatter_plot(data, column_name)
    fig_pie_plot = create_pie_plot(data, column_name)
    fig_bar_plot = create_bar_plot(data, column_name)

    figures = [fig_histogram, fig_box_plot, fig_pie_plot, fig_bar_plot]
    return figures, fig_scatter_plot  


# In[48]:


if __name__ == '__main__':
    # Get user input for file path and file type
    file_path = "International_Report_Departures.csv"
    # Load and preprocess data
    data = read_file(file_path)
    preprocessed_data = preprocess_data(data)

    # Print all features
    print("All Features:")
    print(preprocessed_data.columns)

    # Choose a specific column to visualize
    column_to_visualize = "Month"

    # Generate all types of plots for the chosen column
    figures, _ = create_visualizations_for_column(preprocessed_data, column_to_visualize)
    months = []
    for month in range(1, 13):
        months.append(dt.datetime(year=1994, month=month, day=1).strftime("%B"))

    fl_per_month = list(df.groupby('Month').count().Year)
    plt.xlabel('Month')
    plt.ylabel('Night Flights')
    plt.xticks(range(1,13), months, rotation='vertical')
    plt.plot(range(1,13), np.array(fl_per_month), '.-')
    plt.title('flights in each month of the year')
    plt.show()
 



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




