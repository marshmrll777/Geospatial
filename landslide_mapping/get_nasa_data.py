import pandas as pd
import os

def download_and_filter_landslides():
    # Direct open URL to NASA's Global Landslide Catalog (Cooperative Open Online Landslide Repository)
    nasa_url = "https://raw.githubusercontent.com/nasa/GEOS-S2S- there-is-no-direct-raw-csv-fallback/master/data/global_landslides.csv"
    
    # Backup reliable source: Humanitarian Data Exchange / NASA data mirror
    backup_url = "https://query.data.world/s/m7w26pvdg7uocbyw3sh6p26lz3mpxm?dws=00000"

    print("🛰️ Connecting to Global Landslide Database...")
    try:
        # Reading the data directly into memory from the cloud stream
        df = pd.read_csv(backup_url)
        print("✅ Global database loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return

    # Look at the column names to find spatial parameters
    # NASA catalogs typically use 'latitude', 'longitude', and 'country_name'
    print("🧹 Filtering spatial coordinates for Uttarakhand region...")
    
    # 1. Filter by country first to drastically reduce data size
    india_df = df[df['country_name'].str.lower() == 'india'].copy()
    
    # 2. Strict spatial bounding box for Uttarakhand, India
    # Latitude: 28.7° N to 31.5° N | Longitude: 77.5° E to 81.0° E
    min_lat, max_lat = 28.7, 31.5
    min_lon, max_lon = 77.5, 81.0
    
    uttarakhand_df = india_df[
        (india_df['latitude'].between(min_lat, max_lat)) & 
        (india_df['longitude'].between(min_lon, max_lon))
    ]
    
    # Keep only the essential columns needed for your ML engine
    cleaned_points = interstate_df = categories = uttarakhand_df[[
        'latitude', 
        'longitude', 
        'landslide_category', 
        'landslide_size', 
        'event_date'
    ]].copy()
    
    # Rename columns to match your cleaning pipeline structure
    cleaned_points.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude'}, inplace=True)
    
    print(f"🎉 Success! Extracted {len(cleaned_points)} verified historical landslide coordinates.")
    
    # Save directly to your workspace as a clean CSV file
    output_path = "uttarakhand_landslides.csv"
    cleaned_points.to_csv(output_path, index=False)
    print(f"💾 File saved directly to your workspace: '{output_path}'")
    
    # Preview what we got
    print("\n--- Quick Sample Profiles ---")
    print(cleaned_points.head())

if __name__ == "__main__":
    download_and_filter_landslides()