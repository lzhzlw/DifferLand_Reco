import pandas as pd
from geopy.distance import geodesic

# Read data
path1 = r'/data1/fyliu/e_soil_carbon_model/code/DifferLand_v3.0/drivers/site_coords.csv'
path2 = r'/data1/fyliu/e_soil_carbon_model/code/DifferLand_v3.0/drivers/som.csv'
points_df = pd.read_csv(path1)  
all_points_df = pd.read_csv(path2)  

# Rename columns for consistency
points_df = points_df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
all_points_df = all_points_df.rename(columns={'Lat': 'latitude', 'Lon': 'longitude', 'soc': 'som'})

# Find closest points and save results
results = []

for _, row in points_df.iterrows():
    point_coords = (row['latitude'], row['longitude'])
    
    # Compute distances to all points in the second DataFrame
    all_points_df['distance'] = all_points_df.apply(
        lambda other: geodesic(
            point_coords,
            (other['latitude'], other['longitude'])
        ).meters,
        axis=1
    )
    
    # Find the closest point
    closest_point = all_points_df.loc[all_points_df['distance'].idxmin()]
    
    # Append result
    results.append({
        'SITE_ID': row['SITE_ID'],
        'lon': row['longitude'],
        'lat': row['latitude'],
        'closest_lon': closest_point['longitude'],
        'closest_lat': closest_point['latitude'],
        'distance_m': closest_point['distance'],
        'som': closest_point['som']
    })

# Save the results as a new DataFrame
output_df = pd.DataFrame(results)
output_df.to_csv('/data1/fyliu/e_soil_carbon_model/code/DifferLand_v3.0/drivers/site_coords_som.csv', index=False)

print("Saving the results!")
