import pandas as pd
import numpy as np
import datetime


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    all_ids = sorted(set(df['id_start']).union(set(df['id_end']))) 

    distance_matrix = pd.DataFrame(np.inf, index=all_ids, columns=all_ids)
    
    np.fill_diagonal(distance_matrix.values, 0)
    
    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[id_start, id_end] = distance
        distance_matrix.at[id_end, id_start] = distance


    for k in all_ids:
        for i in all_ids:
            for j in all_ids:
                distance_matrix.at[i, j] = min(distance_matrix.at[i, j], distance_matrix.at[i, k] + distance_matrix.at[k, j])

    return distance_matrix



def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    unrolled_data = []

    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:
                distance = df.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    ref_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

    lower_bound = ref_avg_distance * 0.9
    upper_bound = ref_avg_distance * 1.1


    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()


    result_df = avg_distances[(avg_distances['distance'] >= lower_bound) & 
                              (avg_distances['distance'] <= upper_bound)]


    result_df = result_df.sort_values(by='id_start').reset_index(drop=True)

    return result_df[['id_start']]


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    

    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate

    return df




def calculate_time_based_toll_rates(df) -> pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    time_ranges_weekday = [
        ("00:00:00", "10:00:00", 0.8),
        ("10:00:00", "18:00:00", 1.2),
        ("18:00:00", "23:59:59", 0.8)
    ]
    
    weekend_discount = 0.7
    
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


    results = []


    for index, row in df.iterrows():
        base_rates = row[['moto', 'car', 'rv', 'bus', 'truck']].values
        id_start = int(row['id_start'])  
        id_end = int(row['id_end'])
        distance = row['distance']

        for day_index, day in enumerate(days_of_week):
            if day in ["Saturday", "Sunday"]:
                discounted_rates = base_rates * weekend_discount
            
                end_day = days_of_week[(day_index + 1) % len(days_of_week)]
                results.append([id_start, id_end, distance, day, "00:00:00", end_day, "23:59:59", *discounted_rates])
            else:
         
                for time_index, (start_time_str, end_time_str, factor) in enumerate(time_ranges_weekday):
                    time_adjusted_rates = base_rates * factor
                 
                    end_day = days_of_week[(day_index + time_index + 1) % len(days_of_week)]
                    results.append([id_start, id_end, distance, day, start_time_str, end_day, end_time_str, *time_adjusted_rates])


    result_df = pd.DataFrame(results, columns=['id_start', 'id_end', 'distance', 'start_day', 'start_time', 'end_day', 'end_time', 'moto', 'car', 'rv', 'bus', 'truck'])
    
 
    vehicle_columns = ['moto', 'car', 'rv', 'bus', 'truck']
    result_df[vehicle_columns] = result_df[vehicle_columns].astype(float).round(2)

    return result_df




df = pd.read_csv('datasets\dataset-2.csv')  

# Q9: Calculate the distance matrix
distance_matrix = calculate_distance_matrix(df)
print("Distance Matrix:")
print(distance_matrix.iloc[:7, :7])  

# Q10: Unroll the distance matrix into a long format
unrolled_df = unroll_distance_matrix(distance_matrix) 
print("\nUnrolled Distance Matrix:")
print(unrolled_df.head(9))  

# Q11: Find IDs within 10% threshold for a reference ID 
reference_id = 1001400
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print(f"\nIDs within 10% Threshold of Reference ID ({reference_id}):")
print(ids_within_threshold.head(10))  

# Q12: Calculate toll rates for vehicles
toll_rate_df = calculate_toll_rate(unrolled_df)
print("\nToll Rates:")
print(toll_rate_df.head(10))

# Q13: Calculate time-based toll rates
time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)
print("\nTime-Based Toll Rates:")
print(time_based_toll_df.head(8))
