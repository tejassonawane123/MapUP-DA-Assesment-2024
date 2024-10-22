#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np


# In[56]:


df = pd.read_csv('dataset-2.csv')


# In[57]:


df.head()


# In[58]:


df.info()


# ## Question 9: Distance Matrix Calculation

# In[59]:


import pandas as pd

def calculate_distance_matrix(file_path):
    # Load the dataset
    df = file_path

    # Create a distance DataFrame with unique IDs as indices and columns
    unique_ids = pd.concat([df['id_start'], df['id_end']]).unique()
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    # Fill the distance matrix with known distances
    for _, row in df.iterrows():
        start_id = row['id_start']
        end_id = row['id_end']
        distance = row['distance']
        
        # Update distances in both directions for symmetry
        distance_matrix.at[start_id, end_id] = distance
        distance_matrix.at[end_id, start_id] = distance

    # Calculate cumulative distances
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                # Check if there is a shorter path through k
                if distance_matrix.at[i, k] + distance_matrix.at[k, j] > 0:  # Avoid updating if no distance exists
                    distance_matrix.at[i, j] = min(distance_matrix.at[i, j] or float('inf'), 
                                                    distance_matrix.at[i, k] + distance_matrix.at[k, j])

    # Set diagonal values to 0 (distance from a location to itself)
    for id_ in unique_ids:
        distance_matrix.at[id_, id_] = 0

    return distance_matrix



# In[60]:


distance_df = calculate_distance_matrix(df)
print(distance_df)


# ## Question 10: Unroll Distance Matrix

# In[61]:


import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Initialize an empty list to hold the rows for the unrolled DataFrame
    unrolled_data = []

    # Iterate over each unique ID in the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Skip if id_start is the same as id_end
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                # Only include rows with a valid distance
                if distance > 0:  # or any condition to filter out unwanted distances
                    unrolled_data.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'distance': distance
                    })

    # Create a new DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)
    
    return unrolled_df


# In[62]:


unrolled_df = unroll_distance_matrix(distance_df)
unrolled_df


# ## Question 11: Finding IDs within Percentage Threshold

# In[63]:


import pandas as pd
import numpy as np

def find_ids_within_ten_percentage_threshold(df, reference_id):
    # Filter the DataFrame for rows where id_start is the reference_id
    reference_data = df[df['id_start'] == reference_id]
    
    # Calculate the average distance for the reference_id
    if reference_data.empty:
        return []  # Return an empty list if no data is found for the reference_id

    average_distance = reference_data['distance'].mean()
    
    # Calculate the 10% threshold
    lower_threshold = average_distance * 0.9
    upper_threshold = average_distance * 1.1

    # Find IDs within the specified threshold
    filtered_ids = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]
    
    # Get unique id_start values from the filtered DataFrame
    unique_ids = filtered_ids['id_start'].unique()
    
    # Sort the result and return it as a list
    sorted_ids = sorted(unique_ids)
    
    return sorted_ids


# In[64]:


reference_id = 1001402  # Replace with your desired reference ID
result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print(result_ids)


# ## Question 12: Calculate Toll Rate

# In[65]:


import pandas as pd

def calculate_toll_rate(df):
    # Define the rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient

    return df


# In[66]:


# Assuming unrolled_df is the output from unroll_distance_matrix
toll_rate_df = calculate_toll_rate(unrolled_df)
toll_rate_df


# ## Question 13: Calculate Time-Based Toll Rates

# In[67]:


import pandas as pd
from datetime import time

def calculate_time_based_toll_rates(df):
    # Define the rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Define discount factors based on time intervals for weekdays and weekends
    weekday_factors = [
        ('00:00:00', '10:00:00', 0.8),
        ('10:00:00', '18:00:00', 1.2),
        ('18:00:00', '23:59:59', 0.8)
    ]
    weekend_factor = 0.7

    # Days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Create a list to hold the results
    results = []

    # Iterate through each row in the input DataFrame
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']

        for day in days_of_week:
            if day in days_of_week[:5]:  # Weekdays (Monday to Friday)
                for start_time_str, end_time_str, factor in weekday_factors:
                    start_time = time.fromisoformat(start_time_str)
                    end_time = time.fromisoformat(end_time_str)

                    # Calculate toll rates for each vehicle type
                    toll_rates = {vehicle: distance * rate * factor for vehicle, rate in rate_coefficients.items()}
                    
                    # Append the result for this day and time range
                    results.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'distance': distance,
                        'start_day': day,
                        'start_time': start_time,
                        'end_day': day,
                        'end_time': end_time,
                        **toll_rates
                    })

            else:  # Weekends (Saturday and Sunday)
                start_time = time(0, 0)
                end_time = time(23, 59)

                # Apply a constant weekend discount factor
                toll_rates = {vehicle: distance * rate * weekend_factor for vehicle, rate in rate_coefficients.items()}

                # Append the result for the entire day
                results.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    **toll_rates
                })

    # Create a DataFrame from the results
    result_df = pd.DataFrame(results)

    return result_df


# In[68]:


# Assuming unrolled_df is the output from unroll_distance_matrix
time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)
time_based_toll_df


# In[ ]:




