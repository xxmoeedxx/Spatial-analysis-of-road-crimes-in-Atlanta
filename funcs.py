import contextily
import geopandas as gpd
import rioxarray
import seaborn
import matplotlib.pyplot as plt
from shapely.geometry import box,LineString
import datetime
import numpy as np
import pandas as pd
import openrouteservice
import folium
from scipy.stats import pearsonr
from IPython.display import display
import osmnx
import statsmodels.api as sm
from pysal.explore import esda
from splot.esda import lisa_cluster,moran_scatterplot
from pysal.lib import weights 

def plot_local_morans_i(grid_gdf, w):
    """
    Calculates and plots Local Moran's I for the 'crime_count' variable in the given GeoDataFrame.

    Parameters:
    - grid_gdf (GeoDataFrame): GeoDataFrame containing grid cells with crime count data.
    - w (weights.W): Spatial weights matrix.
    """
    # Calculate Local Moran's I
    local_moran = esda.moran.Moran_Local(grid_gdf['crime_count'], w)

    # Plot Local Moran's I
    lisa_cluster(local_moran, grid_gdf, p=0.05, figsize=(10, 10))
    plt.title("Moran Cluster Map")
    plt.show()


 
def filter_crime_data(gdf):
    """
    Filters crime data GeoDataFrame by city and relevant crime types.

    Parameters:
    - gdf (GeoDataFrame): GeoDataFrame containing crime data.
    
    Returns:
    - filtered_gdf (GeoDataFrame): Filtered GeoDataFrame containing rows with relevant crime types in the specified city.
    """
    # Define the relevant crime types
    relevant_crime_types = ['LARCENY-FROM VEHICLE', 'AUTO THEFT']
    # Geocode the city name to get its geometry
    city_gdf = osmnx.geocode_to_gdf("Atlanta, Georgia, USA")
    city_geom = city_gdf.geometry.iloc[0]

    # Filter rows based on intersection with city geometry
    gdf = gdf[gdf.intersects(city_geom)]

    # Create a boolean mask to filter rows with relevant crime types
    mask = gdf['Crime_Type'].isin(relevant_crime_types)

    # Apply the mask to select rows with relevant crime types
    filtered_gdf = gdf[mask]

    return filtered_gdf


 
def preprocess_date_column(gdf_with_date):
    """
    Preprocesses the 'Occur_Date' column in the GeoDataFrame.

    Parameters:
    - gdf_with_date (GeoDataFrame): GeoDataFrame containing the 'Occur_Date' column.

    Returns:
    - preprocessed_gdf (GeoDataFrame): Preprocessed GeoDataFrame with 'Occur_Date' column processed.
    """
    # Replace "NULL" with NaN
    gdf_with_date['Occur_Date'] = gdf_with_date['Occur_Date'].replace("NULL", np.nan)

    # Convert 'Occur_Date' to datetime format
    gdf_with_date['Occur_Date'] = pd.to_datetime(gdf_with_date['Occur_Date'], errors='coerce', format='%m/%d/%Y')

    # Remove rows with NaN values in 'Occur_Date'
    preprocessed_gdf = gdf_with_date.dropna(subset=['Occur_Date'])

    return preprocessed_gdf



 
def preprocess_time_column(gdf_with_time):
    """
    Preprocesses the 'Occur_Time' column in the GeoDataFrame.

    Parameters:
    - gdf_with_time (GeoDataFrame): GeoDataFrame containing the 'Occur_Time' column.

    Returns:
    - preprocessed_gdf (GeoDataFrame): Preprocessed GeoDataFrame with 'Occur_Time' column processed.
    """
    # Replace "NULL" with NaN
    gdf_with_time['Occur_Time'] = gdf_with_time['Occur_Time'].replace("NULL", np.nan)

    # Convert 'Occur_Time' to datetime format
    gdf_with_time['Occur_Time'] = pd.to_datetime(gdf_with_time['Occur_Time'], errors='coerce', format='%H:%M')

    # Remove rows with NaN values in 'Occur_Time'
    preprocessed_gdf = gdf_with_time.dropna(subset=['Occur_Time'])

    return preprocessed_gdf


 
def load_population_data(years, pop_file_paths, study_area_bounds):
    """
    Loads population data from GeoTIFF files for specified years, clips them to the bounding box of the study area,
    and stores them in a dictionary.

    Parameters:
    - years (list): List of years for which population data is to be loaded.
    - pop_file_paths (dict): Dictionary containing file paths for each year's population data.
    - study_area_bounds (tuple): Tuple containing the bounding box coordinates (xmin, ymin, xmax, ymax) of the study area.

    Returns:
    - pops (dict): Dictionary containing loaded and clipped population data arrays for each year.
    """
    pops = {}

    for year in years:
        # Load the population data
        pop_file_path = pop_file_paths.get(year)
        if not pop_file_path:
            print(f"Population data file path not found for year {year}. Skipping...")
            continue
        
        pop = rioxarray.open_rasterio(pop_file_path)
        
        # Clip the population data to the bounding box of the study area
        pop_clipped = pop.rio.clip_box(*study_area_bounds)
        
        # Add the clipped population data to the pops dictionary
        pops[year] = pop_clipped
    
    return pops



 
def plot_crime_distribution_by_neighborhood(crime_gdf):
    """
    Plots the distribution of crimes by neighborhood.

    Parameters:
    - crime_gdf (GeoDataFrame or DataFrame): DataFrame containing crime data with a 'Neighborhood' column.
    """
    # Replace "NULL" values with NaN and drop rows with NaN in 'Neighborhood' column
    crime_gdf['Neighborhood'] = crime_gdf['Neighborhood'].replace("NULL", np.nan)
    crime_gdf = crime_gdf.dropna(subset=['Neighborhood'])

    # Group the data by 'Neighborhood' and count the occurrences
    crime_counts = crime_gdf.groupby('Neighborhood').size()

    # Get top 25 neighborhoods with the most crimes
    top_25_crimes = crime_counts.sort_values(ascending=False).head(25)

    # Plot the distribution chart
    plt.figure(figsize=(12, 6))
    top_25_crimes.plot(kind='bar')
    plt.title('Distribution of Crimes by Neighborhood')
    plt.xlabel('Neighborhood')
    plt.ylabel('Number of Crimes')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


 
def plot_crime_months_distribution(crime_gdf):
    """
    Plots the distribution of crimes by month.

    Parameters:
    - crime_gdf (GeoDataFrame or DataFrame): DataFrame containing crime data with an 'Occur_Date' column.
    """
    # Define month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    # Extract month from 'Occur_Date'
    months = crime_gdf['Occur_Date'].dt.month_name()

    # Count crimes per month
    monthly_crimes = months.value_counts()

    # Reindex monthly_crimes with desired month order
    monthly_crimes = monthly_crimes.reindex(month_order, fill_value=0)

    # Create the bar chart
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.bar(monthly_crimes.index, monthly_crimes.values)
    plt.xlabel('Month')
    plt.ylabel('Number of Crimes')
    plt.title('Crimes by Month')
    plt.xticks(rotation=45, ha='right')  # Rotate month labels for better readability
    plt.tight_layout()
    plt.show()

def plot_crime_distribution_by_years(gdf):
    """
    This function plots the distribution of crimes by years.
    """
    # Create a new column for the year of the crime
    gdf['Year'] = gdf['Occur_Date'].dt.year
    gdf = gdf[gdf['Year'] >= 2009]

    # Group the data by year and count the number of crimes in each year
    crime_counts = gdf.groupby('Year').size()

    # Plot the distribution of crimes by years
    ax = crime_counts.plot(kind='bar', figsize=(10, 6), color='skyblue', edgecolor='black')

    # Set the title and labels
    ax.set_title('Distribution of Crimes by Years', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Number of Crimes', fontsize=14)

    # Display the plot
    plt.show()





 
def plot_crime_hours_distribution(crime_gdf):
    """
    Plots the distribution of crimes by hour.

    Parameters:
    - crime_gdf (GeoDataFrame or DataFrame): DataFrame containing crime data with an 'Occur_Time' column.
    """
    # Extract hour from 'Occur_Time'
    hours = crime_gdf['Occur_Time'].hour

    # Count crimes per hour
    hourly_crimes = hours.value_counts()

    # Sort the index to display hours in order
    hourly_crimes = hourly_crimes.sort_index()

    # Create the bar chart
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.bar(hourly_crimes.index, hourly_crimes.values)
    plt.xlabel('Hour')
    plt.ylabel('Number of Crimes')
    plt.title('Crimes by Hour')
    plt.xticks(range(24))  # Display all 24 hours
    plt.tight_layout()
    plt.show()


 
def plot_crime_types_pie_chart(crime_gdf):
    """
    Plots a pie chart showing the distribution of crime types.

    Parameters:
    - crime_gdf (GeoDataFrame or DataFrame): DataFrame containing crime data with a 'Crime_Type' column.
    """
    # Count the occurrences of each crime type
    crime_type_counts = crime_gdf['Crime_Type'].value_counts()

    # Plot the distribution chart
    plt.figure(figsize=(12, 6))
    crime_type_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Crime Types')
    plt.ylabel('')
    plt.show()


 
def get_grid_parameters(atl, pops, year):
    """
    Calculates the bounding box and cell size for a grid based on a given study area and population data.

    Parameters:
    - atl (GeoDataFrame): GeoDataFrame representing the study area.
    - pops (dict of xarray.DataArray): Dictionary of population density data, keyed by year.
    - year (int): The year to consider for population data.

    Returns:
    - xmin, ymin, xmax, ymax (float): The bounding box coordinates of the study area.
    - cell_size (float): The size of the grid cells.
    """
    xmin, ymin, xmax, ymax = atl.total_bounds
    cell_size_x = np.abs(pops[year]['x'][1] - pops[year]['x'][0])
    cell_size_y = np.abs(pops[year]['y'][1] - pops[year]['y'][0])

    # Ensure cell size is consistent (could be square or rectangular cells)
    assert np.isclose(cell_size_x, cell_size_y), "Cell sizes in x and y direction are not equal."
    cell_size = cell_size_x  # or cell_size_y (adjust as needed)
    
    return xmin, ymin, xmax, ymax, cell_size



 
def generate_yearly_grids_with_population(pops, gdf_with_date, xmin, ymin, xmax, ymax, cell_size, starting_year=2016, ending_year=2020):
    """
    Processes grid data for a specified geographic area over a range of years, adding population density and crime count data.

    Parameters:
    - pops (dict of xarray.DataArray): Dictionary of population density data, keyed by year.
    - gdf_with_date (GeoDataFrame): GeoDataFrame containing crime data with a date column.
    - xmin, ymin, xmax, ymax (float): The bounding box coordinates of the study area.
    - cell_size (float): The size of the grid cells.
    - starting_year (int): The starting year for processing the data (default is 2016).
    - ending_year (int): The ending year for processing the data (default is 2020).

    Returns:
    - grid_gdfs (dict of GeoDataFrame): Dictionary of processed GeoDataFrames, keyed by year.
      Each GeoDataFrame contains grid cells with population density and crime count data.
    """
    grid_gdfs = {}
    

    for year in range(starting_year, ending_year + 1):
        # Create a grid of polygons covering the study area
        grid_cells = []
        for x in np.arange(xmin, xmax, cell_size):
            for y in np.arange(ymin, ymax, cell_size):
                cell_box = box(x, y, x + cell_size, y + cell_size)
                grid_cells.append(cell_box)

        # Create a GeoDataFrame for the grid cells
        grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=gdf_with_date.crs)

        # Compute centroids of the grid cells
        grid_gdf['centroid'] = grid_gdf.centroid

        # Extract centroid coordinates
        centroid_coords = np.array([(geom.x, geom.y) for geom in grid_gdf['centroid']])

        # Create DataFrame with centroid coordinates
        centroid_df = pd.DataFrame(centroid_coords, columns=['x', 'y'])

        # Sample population density values from pop_clipped using xarray
        pop_values = []
        for _, row in centroid_df.iterrows():
            pop_value = pops[year].sel(x=row['x'], y=row['y'], method="nearest").values
            pop_values.append(pop_value)

        # Add population density values to grid_gdf
        grid_gdf['population_density'] = pop_values

        # Filter crime_gdf for the current year
        start_date = pd.to_datetime(f'{year}-01-01')
        end_date = pd.to_datetime(f'{year}-12-31')
        crime_gdf2 = gdf_with_date[(gdf_with_date['Occur_Date'] >= start_date) & (gdf_with_date['Occur_Date'] <= end_date)].dropna()

        # Perform spatial join to count crime incidents within each grid cell
        joined_gdf = gpd.sjoin(crime_gdf2, grid_gdf, how="right", op='within')

        # Group by grid cells and count the number of crime incidents in each cell
        crime_counts = joined_gdf.groupby(level=0).size().reset_index(name='crime_count')

        # Merge crime counts with grid data
        grid_gdf = pd.merge(grid_gdf, crime_counts, how='left', left_index=True, right_on='index')

        # Fill missing values (cells with no crime incidents) with 0
        grid_gdf['crime_count'].fillna(0, inplace=True)

        # Filter out rows with crime_count <= 1
        grid_gdf = grid_gdf[grid_gdf['crime_count'] > 1].reset_index(drop=True)

        # Store the processed GeoDataFrame for the current year
        grid_gdfs[year] = grid_gdf
    
    for year in range(starting_year, ending_year + 1):
        grid_gdfs[year]['population_density'] = grid_gdf['population_density'].apply(lambda x: float(x) if isinstance(x, (int, float, np.number)) else float(x[0]) if isinstance(x, np.ndarray) else np.nan)

    return grid_gdfs


 
def plot_crime_and_pop_for_year(grid_gdfs, year):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

    # Plot crime counts
    grid_gdfs[year].plot(column='crime_count', scheme='fisher_jenks', cmap='Blues', legend=True, ax=ax1)
    contextily.add_basemap(ax1, crs=grid_gdfs[year].crs, source=contextily.providers.OpenStreetMap.Mapnik)
    ax1.set_title('Crime Counts in Grid Cells ({year})')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    # Add crime counts as annotations
    for _, row in grid_gdfs[year].iterrows():
        ax1.text(row['geometry'].centroid.x, row['geometry'].centroid.y, int(row['crime_count']),
                horizontalalignment='center', verticalalignment='center', fontsize=6, color='black')

    # Plot population density
    grid_gdfs[year].plot(column='population_density',scheme='fisher_jenks', cmap='Reds', legend=True, ax=ax2)
    contextily.add_basemap(ax2, crs=grid_gdfs[year].crs, source=contextily.providers.OpenStreetMap.Mapnik)
    ax2.set_title('Population Density in Grid Cells ({year})')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    # Add population density as annotations
    for _, row in grid_gdfs[year].iterrows():
        ax2.text(row['geometry'].centroid.x, row['geometry'].centroid.y, f"{row['population_density']:.2f}",
                horizontalalignment='center', verticalalignment='center', fontsize=6, color='black')

    # Show the plots
    plt.tight_layout()
    plt.show()


 
def calculate_correlation_for_each_year(grid_gdfs):
    """
    Calculates the Pearson correlation coefficient between crime counts and population density
    for each year in the grid_gdfs dictionary.

    Parameters:
    - grid_gdfs (dict): Dictionary containing GeoDataFrames of grid cells with crime count and population density data.

    Returns:
    - correlations (dict): Dictionary containing Pearson correlation coefficients and p-values for each year.
    """
    correlations = {}

    for year, grid_gdf in grid_gdfs.items():
        # Extract 'crime_count' and 'population_density' columns
        crime_counts = grid_gdf['crime_count']
        population_density = grid_gdf['population_density']

        # Combine the two series into a DataFrame
        df = pd.DataFrame({'crime_count': crime_counts, 'population_density': population_density})

        # Drop rows with NaN or infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Extract the cleaned data
        crime_counts_clean = df['crime_count']
        population_density_clean = df['population_density']

        # Calculate Pearson correlation
        correlation, p_value = pearsonr(crime_counts_clean, population_density_clean)

        # Store the results
        correlations[year] = {'correlation': correlation, 'p_value': p_value}

        # Print the results
        print(f"Year: {year}")
        print(f"Pearson correlation coefficient: {correlation:.4f}")
        print(f"P-value: {p_value:.4e}\n")

    return correlations



 
def calculate_combined_correlation(grid_gdfs):
    """
    Calculates the Pearson correlation coefficient between crime counts and population density
    across all years in the grid_gdfs dictionary.

    Parameters:
    - grid_gdfs (dict): Dictionary containing GeoDataFrames of grid cells with crime count and population density data
                        for different years.

    Returns:
    - correlation (float): Pearson correlation coefficient for combined data across all years.
    - p_value (float): P-value for the correlation.
    """
    # Combine crime counts and population density across years
    crime_counts = []
    population_density = []
    for year in range(2016, 2021):
        crime_counts.extend(grid_gdfs[year]['crime_count'].tolist())
        population_density.extend(grid_gdfs[year]['population_density'].tolist())

    # Combine data into arrays
    crime_counts_all = np.array(crime_counts)
    population_density_all = np.array(population_density)

    # Handle missing values (replace with NaN or use appropriate method)
    crime_counts_all = np.nan_to_num(crime_counts_all)  # Replaces with NaN
    population_density_all = np.nan_to_num(population_density_all)

    # Calculate Pearson correlation
    correlation, p_value = pearsonr(crime_counts_all, population_density_all)

    # Print the results
    print("Pearson correlation coefficient (all years):", correlation.round(4))
    print(f"P-value (all years): {p_value:.4e}")

    return correlation, p_value



 
def perform_linear_regression_yearly(grid_gdfs):
    """
    Performs linear regression analysis between crime counts and population density
    for each year in the grid_gdfs dictionary.

    Parameters:
    - grid_gdfs (dict): Dictionary containing GeoDataFrames of grid cells with crime count and population density data
                        for different years.

    Returns:
    - regression_results (dict): Dictionary containing regression results summary for each year.
    """
    regression_results = {}

    for year in range(2016, 2021):
        # Remove rows with missing values
        crime_counts = grid_gdfs[year]['crime_count']
        population_density = grid_gdfs[year]['population_density']

        data = pd.DataFrame({'crime_count': crime_counts, 'population_density': population_density}).dropna()

        # Add a constant term for the intercept
        X = sm.add_constant(data['population_density'])
        y = data['crime_count']

        # Fit the linear regression model
        model = sm.OLS(y, X).fit()

        # Store the summary of the regression results
        regression_results[year] = model.summary()

        # Print the summary of the regression results
        print(f"Year: {year}")
        print(model.summary())

    return regression_results


 
def perform_combined_linear_regression(grid_gdfs):
    """
    Performs linear regression analysis between crime counts and population density
    across all years in the grid_gdfs dictionary.

    Parameters:
    - grid_gdfs (dict): Dictionary containing GeoDataFrames of grid cells with crime count and population density data
                        for different years.

    Returns:
    - regression_results_summary (str): Summary of the regression results for all years combined.
    """
    # Combine crime counts and population density across years
    crime_counts = []
    population_density = []
    for year in range(2016, 2021):
        crime_counts.extend(grid_gdfs[year]['crime_count'].tolist())
        population_density.extend(grid_gdfs[year]['population_density'].tolist())

    # Create a DataFrame with combined data
    data = pd.DataFrame({'crime_count': crime_counts, 'population_density': population_density})

    # Handle missing values (you can choose to drop rows or impute)
    data.dropna(inplace=True)  # Drops rows with missing values

    # Add a constant term for the intercept
    X = sm.add_constant(data['population_density'])
    y = data['crime_count']

    # Fit the linear regression model
    model = sm.OLS(y, X).fit()

    # Get the summary of the regression results
    regression_results_summary = model.summary()

    # Print the summary of the regression results
    print("Regression results for all years combined:")
    print(regression_results_summary)

    return regression_results_summary


 
def plot_crime_and_boundary(crime_gdf, boundary_gdf):
    """
    Plots crime incidents and the boundary of the study area, overlaying a basemap.

    Parameters:
    - crime_gdf (GeoDataFrame): GeoDataFrame containing crime data with geometry.
    - boundary_gdf (GeoDataFrame): GeoDataFrame representing the boundary of the study area.
    """
    fig, ax = plt.subplots(1, figsize=(9, 9))
    
    # Plot crime incidents
    crime_gdf.plot(ax=ax, color="red", markersize=1)
    
    # Plot the boundary
    boundary_gdf.boundary.plot(ax=ax, color="black")
    
    # Add basemap
    contextily.add_basemap(ax, crs=crime_gdf.crs.to_string(), source=contextily.providers.OpenStreetMap.Mapnik)
    
    # Show plot
    plt.show()



 
def plot_kde_with_basemap(crime_gdf):
    """
    Creates a KDE plot of crime incidents and overlays it on a basemap.

    Parameters:
    - crime_gdf (GeoDataFrame): GeoDataFrame containing crime data with geometry.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot KDE
    seaborn.kdeplot(
        x=crime_gdf.geometry.x,
        y=crime_gdf.geometry.y,
        n_levels=50,
        shade=True,
        alpha=0.55,
        cmap="viridis_r",
        ax=ax
    )

    # Add basemap
    contextily.add_basemap(
        ax, crs=crime_gdf.crs.to_string(), source=contextily.providers.OpenStreetMap.Mapnik
    )

    # Remove axes
    ax.set_axis_off()

    # Show plot
    plt.show()



 
def create_grid_and_count_crimes(crime_gdf, xmin, ymin, xmax, ymax, cell_size):
    """
    Creates a grid of polygons covering the study area, performs a spatial join to count 
    crime incidents within each grid cell, and processes the data.

    Parameters:
    - crime_gdf (GeoDataFrame): GeoDataFrame containing crime data with a date column.
    - xmin, ymin, xmax, ymax (float): The bounding box coordinates of the study area.
    - cell_size (float): The size of the grid cells.

    Returns:
    - grid_gdf (GeoDataFrame): GeoDataFrame containing grid cells with crime count data.
    """
    # Create a grid of polygons covering the study area
    grid_cells = []
    for x in np.arange(xmin, xmax, cell_size):
        for y in np.arange(ymin, ymax, cell_size):
            cell_box = box(x, y, x + cell_size, y + cell_size)
            grid_cells.append(cell_box)

    # Create a GeoDataFrame for the grid cells
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=crime_gdf.crs)

    # Perform spatial join to count crime incidents within each grid cell
    joined_gdf = gpd.sjoin(crime_gdf, grid_gdf, how="right", op='within')

    # Group by grid cells and count the number of crime incidents in each cell
    crime_counts = joined_gdf.groupby(level=0).size().reset_index(name='crime_count')

    # Merge crime counts with grid data
    grid_gdf = pd.merge(grid_gdf, crime_counts, how='left', left_index=True, right_on='index')

    # Fill missing values (cells with no crime incidents) with 0
    grid_gdf['crime_count'].fillna(0, inplace=True)

    # Filter out rows with crime_count <= 1
    grid_gdf = grid_gdf[grid_gdf['crime_count'] > 1].reset_index(drop=True)

    return grid_gdf


 
def plot_crime_counts(grid_gdf):
    """
    Plots the grid cells with crime counts written on them.

    Parameters:
    - grid_gdf (GeoDataFrame): GeoDataFrame containing grid cells with crime count data.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    grid_gdf.plot(column='crime_count', scheme='fisher_jenks', cmap='Blues', legend=True, ax=ax)
    contextily.add_basemap(ax, crs=grid_gdf.crs, source=contextily.providers.OpenStreetMap.Mapnik)

    # Add crime counts as annotations
    for _, row in grid_gdf.iterrows():
        ax.text(row['geometry'].centroid.x, row['geometry'].centroid.y, int(row['crime_count']),
                horizontalalignment='center', verticalalignment='center', fontsize=6, color='black')

    plt.title('Crime Counts in Grid Cells')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


 
def compute_morans_i(grid_gdf, w):
    """
    Calculates Moran's I for the 'crime_count' variable in a GeoDataFrame.

    Parameters:
    - grid_gdf (GeoDataFrame): GeoDataFrame containing grid cells with crime count data.
    - w (weights.W): Spatial weights matrix.

    Returns:
    - moran (esda.moran.Moran): Moran's I statistic object containing the results.
    """

    # Create a spatial lag of the 'crime_count' variable
    grid_gdf['lag_crime_count'] = weights.lag_spatial(w, grid_gdf['crime_count'])

    # Calculate Moran's I
    moran = esda.moran.Moran(grid_gdf['crime_count'], w)
    
    return moran


 
def plot_local_morans_i_kde(local_moran):
    """
    Draws a Kernel Density Estimate (KDE) plot and adds a rug plot for the Local Moran's I values.

    Parameters:
    - local_moran (esda.moran.Moran_Local): Local Moran's I statistic object containing the results.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw KDE line
    seaborn.kdeplot(local_moran.Is, shade=True, ax=ax)
    
    # Add one small bar (rug) for each observation along horizontal axis
    seaborn.rugplot(local_moran.Is, ax=ax)

    ax.set_title("KDE and Rug Plot of Local Moran's I Values")
    ax.set_xlabel("Local Moran's I")
    ax.set_ylabel("Density")
    plt.show()


 
def calculate_safest_route(coords1, coords2, gdf_with_time, num_alternatives=2, buffer_distance_meters=100):
    """
    Calculates the safest route among multiple alternatives using ORS.

    Args:
        coords1 (tuple): Origin coordinates (latitude, longitude).
        coords2 (tuple): Destination coordinates (latitude, longitude).
        crime_gdf (geopandas.GeoDataFrame): Crime data as a GeoDataFrame.
        num_alternatives (int, optional): Number of alternative routes to consider (default: 2).
        buffer_distance (float, optional): Buffer distance in meters to account for proximity (default: 100).

    Returns:
        tuple: A tuple containing the safest route geometry and its safety score, or None if no routes found.
    """

    api_key = '5b3ce3597851110001cf6248a498d6d0af634364a69375bff5931e0c'  # Replace with your ORS API key
    client = openrouteservice.Client(key=api_key)
    #filter crime gdf with respect to crime_gdf["Occur_Time"] so it is looks for half hour behind and ahead of current time
        
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    print("Current date and time:", current_datetime.strftime("%H:%M"))
    # Define the time difference as a timedelta
    diff_time = datetime.timedelta(minutes=30)

    # Create the datetime range
    start_datetime = current_datetime - diff_time
    end_datetime = current_datetime + diff_time

    # Extract the time component from the datetime objects
    start_time = start_datetime.time()
    end_time = end_datetime.time()
    
    print("Looking for all crimes that occurred between", start_datetime.strftime("%H:%M"), "and", end_datetime.strftime("%H:%M"))

    # Assuming 'Occur_Time' is of type datetime.time in the crime_gdf DataFrame
    if start_time > end_time:
        # Create a mask for the time range that spans midnight
        mask = (gdf_with_time["Occur_Time"].dt.time >= start_time) | (gdf_with_time["Occur_Time"].dt.time <= end_time)
        
    else:
        # Create a mask for the time range within the same day
        mask = (gdf_with_time["Occur_Time"].dt.time >= start_time) & (gdf_with_time["Occur_Time"].dt.time <= end_time)
    
    crime_gdf2 = gdf_with_time[mask]


    # Reproject crime data to the same CRS as ORS (e.g., UTM)
    crime_gdf_utm = crime_gdf2.to_crs(epsg=32633)
    safest_route = None
    min_score = float('inf')

    # Loop to retrieve and evaluate multiple alternatives


    try:
        routes = client.directions(
            coordinates=[coords1, coords2],
            format='geojson',
            alternative_routes={"target_count":3}
        )

        if routes['features']:
            for i in range(num_alternatives):
                route_geometry = LineString(routes['features'][i]['geometry']['coordinates'])
                
                # Buffer the route for proximity analysis
                route_gdf = gpd.GeoDataFrame(geometry=[route_geometry], crs="EPSG:4326")
                route_gdf_utm = route_gdf.to_crs(epsg=32633)
                
                route_buffer = route_gdf_utm.buffer(buffer_distance_meters)
                buffer_gdf = gpd.GeoDataFrame(geometry=route_buffer, crs=route_gdf_utm.crs)
                
                # Find intersecting crimes
                intersecting_crimes = crime_gdf_utm[crime_gdf_utm.intersects(buffer_gdf.unary_union)]
                safety_score = len(intersecting_crimes)

                print(f"Found {safety_score} intersecting crimes for route {i+1}.")

                if safety_score < min_score:
                    safest_route = route_geometry
                    min_score = safety_score
    except openrouteservice.exceptions.ORSRequestError as e:
        print(f"Error retrieving routes from ORS: {e}")

    return safest_route, min_score




 
def visualize_safest_route(safest_route, safety_score, map_center=[33.748992, -84.390264]):
    """
    Visualizes the safest route on a Folium map.

    Parameters:
    - safest_route (LineString): LineString representing the safest route.
    - safety_score (float): Safety score of the route.
    - map_center (list): Optional, list containing latitude and longitude of the map center. Default is Atlanta.

    Returns:
    - None
    """
    # Reverse the coordinates to match Folium's expected format
    reversed_coords = [(lat, lon) for lon, lat in safest_route.coords]
    safest_route = LineString(list(reversed_coords))
    
    # Create a Folium map centered around the specified location
    m = folium.Map(location=map_center, zoom_start=12)

    # Add the safest route as a Polyline to the map
    folium.PolyLine(locations=list(safest_route.coords), color='green', weight=5).add_to(m)

    # Print the safety score
    print("Choosing route with safety score:", safety_score)

    # Display the map
    display(m)


