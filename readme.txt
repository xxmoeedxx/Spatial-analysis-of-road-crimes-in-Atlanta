 Overview
This project analyzes the relationship between population density and crime counts in Atlanta, Georgia, from 2009 to 2020. It uses various data processing, visualization, and statistical techniques to understand spatial and temporal patterns in crime data.

 Prerequisites
Ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)

 Libraries
The following Python libraries are required for this project:
- `geopandas`
- `osmnx`
- `pysal`
- `splot`
- `matplotlib`
- `seaborn`
- `openrouteservice`
- `contextily`
- `rioxarray`
- `shapely`
- `scipy`
- `numpy`
- `pandas`
- `folium`
- `statsmodels`

You can install these libraries using the following command:
"pip install geopandas osmnx pysal splot matplotlib seaborn openrouteservice contextily rioxarray shapely scipy numpy pandas folium statsmodels"

 Installation
1. Clone this repository to your local machine.
2. Navigate to the project directory.

 Data Preparation
1. Ensure the crime data (`2009_2020CrimeData.geojson`) and population density data (TIFF files) are in the `Dataset` directory.
2. The data should be structured as follows:
   - `Dataset/2009_2020CrimeData.geojson`
   - `Dataset/usa_pd_2016_1km.tif`
   - `Dataset/usa_pd_2017_1km.tif`
   - `Dataset/usa_pd_2018_1km.tif`
   - `Dataset/usa_pd_2019_1km.tif`
   - `Dataset/usa_pd_2020_1km.tif`

 Running the Code
1. Open the Jupyter Notebook (`notebook.ipynb`) in Jupyter Lab or Jupyter Notebook.
2. Run all cells sequentially to execute the code.

 Key Libraries and Their Usage
 `geopandas`
- Used for reading and processing geospatial data. It extends pandas to allow spatial operations on geometric types.

 `osmnx`
- Utilized for retrieving and working with street networks. In this project, it helps define the bounding box for Atlanta.

 `pysal` and `splot`
- `pysal` is used for spatial data analysis, including the creation of spatial weights and spatial regression.
- `splot` provides visualization tools for spatial data, such as LISA (Local Indicators of Spatial Association) cluster maps.

 `matplotlib` and `seaborn`
- These libraries are used for data visualization. `matplotlib` provides the foundation for plotting, while `seaborn` offers advanced visualization tools to make plots more informative.

 `openrouteservice`
- This library interacts with the OpenRouteService API to find the safest routes based on crime data. Make sure to configure your API key before using this functionality.

 `contextily`
- Used for adding basemaps to plots, which helps in visualizing geospatial data over map backgrounds.

 `rioxarray`
- Facilitates the reading and processing of raster data (e.g., TIFF files for population density).

 `shapely`
- Provides geometric objects and operations for spatial analysis.

 `scipy`
- Used for statistical analysis, such as calculating Pearson correlation coefficients.

 `numpy` and `pandas`
- `numpy` is used for numerical operations and array manipulations.
- `pandas` is used for data manipulation and analysis.

 `folium`
- Used for creating interactive maps to visualize routes and spatial data.

 `statsmodels`
- Provides tools for estimating and testing statistical models, such as linear regression.

 Key Functions
 `load_population_data(years, file_paths, bbox)`
Loads population density data for specified years and clips it to the provided bounding box.

 `filter_crime_data(gdf)`
Filters the crime data to retain only necessary columns.

 `preprocess_date_column(gdf)`
Formats the date column for temporal analysis.

 `preprocess_time_column(gdf)`
Formats the time column for temporal analysis.

 `generate_yearly_grids_with_population(pops, gdf, xmin, ymin, xmax, ymax, cell_size)`
Generates spatial grids for each year and aggregates population and crime data within these grids.

 `calculate_correlation_for_each_year(grids)`
Calculates the Pearson correlation coefficient between population density and crime counts for each year.

 `perform_linear_regression_yearly(grids)`
Performs linear regression analysis for each year to understand the relationship between population density and crime counts.

 `calculate_safest_route(coords1, coords2, gdf, num_alternatives)`
Finds and visualizes the safest route between two coordinates based on crime data.

 Results
 Analysis Outputs
- Crime distribution by neighborhood, month, and time.
- Correlation between population density and crime counts.
- Spatial regression results.
- Spatial autocorrelation and hotspot analysis.

 Visualizations
- Crime distribution maps.
- Population density and crime density overlays.
- Correlation and regression plots.
- Spatial autocorrelation plots.

 Conclusion
This project provides insights into the spatial and temporal patterns of crime in Atlanta, helping identify potential correlations between population density and crime. The analysis results and visualizations can aid in urban planning, law enforcement strategies, and community safety initiatives.

For more details, refer to the full code and comments in `notebook.ipynb`.

 Contact
For any questions or issues, please contact Moeed Ahmad at moeedahmed254@gmail.com .