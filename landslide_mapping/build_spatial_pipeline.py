import os
import pickle
import numpy as np
import pandas as pd

def step_1_extract_raw_historical_coordinates():
    """
    Simulates sourcing historical landslide coordinates (Label 1) 
    within the geographic bounding box of Uttarakhand.
    """
    print("🛰️ [STEP 1/4] Extracting historical ground-truth coordinates...")
    np.random.seed(101)
    
    # Bounding Box for Uttarakhand: Lat (28.7 to 31.5), Lon (77.5 to 81.0)
    total_incidents = 450
    lat_landslides = np.random.uniform(29.5, 31.2, total_incidents)
    lon_landslides = np.random.uniform(78.2, 80.5, total_incidents)
    
    # Historical events classification
    categories = np.random.choice(['Mudslide', 'Rockfall', 'Debris Flow'], total_incidents)
    
    raw_df = pd.DataFrame({
        'latitude': lat_landslides,
        'longitude': lon_landslides,
        'incident_type': categories,
        'source_agency': 'Sourced-Local-Grid'
    })
    
    # Save raw baseline artifact to workspace
    raw_df.to_csv("uttarakhand_landslides.csv", index=False)
    print("  -> Raw historical coordinate matrix generated successfully.")


def step_2_clean_and_align_features():
    """
    Reads the coordinate file, builds contrast pseudo-absence points (Label 0),
    extracts spatial terrain/InSAR layers, and handles missing sensor parameters.
    """
    print("\n🧹 [STEP 2/4] Initializing alignment & spatial data cleaning...")
    
    # Load coordinates
    coords_df = pd.read_csv("uttarakhand_landslides.csv")
    num_points = len(coords_df)
    
    # Core coordinates (Positive Class = 1)
    pos_lat = coords_df['latitude'].values
    pos_lon = coords_df['longitude'].values
    
    # Generate Pseudo-Absence Points (Negative Class = 0: stable areas)
    neg_lat = np.random.uniform(28.8, 29.8, num_points)
    neg_lon = np.random.uniform(77.6, 78.5, num_points)
    
    # Combine spatial coordinate pools
    all_lats = np.concatenate([pos_lat, neg_lat])
    all_lons = np.concatenate([pos_lon, neg_lon])
    labels = np.concatenate([np.ones(num_points), np.zeros(num_points)])
    
    # Extracting core remote sensing values (Warping layers to fit points)
    # Steep slopes correlate with landslide points; lower slopes with stable points
    slope_noise = np.random.normal(0, 5, num_points * 2)
    slopes = np.concatenate([np.random.uniform(25, 52, num_points), np.random.uniform(2, 18, num_points)]) + slope_noise
    
    # InSAR Ground displacement velocity (Negative values = active subsidence/downward movement)
    insar_noise = np.random.normal(0, 2, num_points * 2)
    insar_disp = np.concatenate([np.random.uniform(-35, -8, num_points), np.random.uniform(-3, 4, num_points)]) + insar_noise
    
    # Soil Moisture saturation percentage
    moisture = np.concatenate([np.random.uniform(40, 85, num_points), np.random.uniform(12, 45, num_points)])
    
    # Consolidating master DataFrame matrix
    master_matrix = pd.DataFrame({
        'Longitude': all_lons,
        'Latitude': all_lats,
        'Slope_Angle': slopes,
        'Soil_Moisture': moisture,
        'InSAR_Displacement': insar_disp,
        'Landslide_Susceptibility': labels.astype(int)
    })
    
    # Injecting standard satellite null artifacts (-9999) to demonstrate cleaning logic
    null_indices = np.random.choice(master_matrix.index, size=25, replace=False)
    master_matrix.loc[null_indices, 'InSAR_Displacement'] = -9999
    
    # DATA PURGE PIPELINE
    print(f"  -> Rows before anomaly purge: {len(master_matrix)}")
    master_matrix = master_matrix.replace(-9999, np.nan)
    master_matrix = master_matrix.dropna().reset_index(drop=True)
    
    # Trim unphysical sensor bounding errors
    master_matrix = master_matrix[(master_matrix['Slope_Angle'].between(0, 90))]
    print(f"  -> Rows after anomaly purge: {len(master_matrix)}")
    
    return master_matrix


def step_3_serialize_to_pickle(cleaned_dataframe):
    """
    Serializes and writes the finalized object directly to a .pkl stream 
    within your local VS Code workspace.
    """
    print("\n📦 [STEP 3/4] Storing processed spatial dataframe to binary pickle file...")
    pkl_filename = "cleaned_geospatial_matrix.pkl"
    
    with open(pkl_filename, "wb") as file:
        pickle.dump(cleaned_dataframe, file)
        
    print(f"  -> Success! Pickle byte file written to: '{os.path.abspath(pkl_filename)}'")


def step_4_verify_pickle_load():
    """
    Sanity test to read the binary file back into memory to ensure it works.
    """
    print("\n🔬 [STEP 4/4] Verifying pickle structural integrity...")
    with open("cleaned_geospatial_matrix.pkl", "rb") as file:
        loaded_df = pickle.load(file)
        
    print("  -> Read verification complete. Previewing matrix stream:")
    print(loaded_df.head(4))


if __name__ == "__main__":
    step_1_extract_raw_historical_coordinates()
    cleaned_data = step_2_clean_and_align_features()
    step_3_serialize_to_pickle(cleaned_data)
    step_4_verify_pickle_load()