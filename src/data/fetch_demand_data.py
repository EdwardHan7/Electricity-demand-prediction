import pandas as pd

def load_demand_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the demand data from a CSV and return it as a DataFrame.
    
    Parameters:
    - csv_path (str): The path to the CSV file.
    
    Returns:
    - DataFrame: The loaded data.
    """
    # Read the CSV file
    data = pd.read_csv(csv_path, 
                       skiprows=4,  # Skip the first four rows
                       thousands=',')  # Handle numbers with commas

    # Parse the date and time
    data['DateTime'] = pd.to_datetime(data['Date'].str.split().str[0] + 
                                      ' ' + data['Date'].str.split().str[1] + 
                                      ':00', 
                                      errors='coerce', 
                                      format='%m/%d/%Y %H:%M')

    # We can assume "Day Ahead Forecast Pool Price" is a float column, but to handle potential string splits, we process it as before
    data['Day Ahead Forecast Pool Price'] = data['Day Ahead Forecast Pool Price'].astype(str).str.split().str[1].astype(float)

    # Drop rows with NaN DateTime
    data.dropna(subset=['DateTime'], inplace=True)

    # Drop columns that are entirely NaN
    data = data.dropna(axis=1, how='all')
    
    return data

# Use the function to load the data
if __name__ == "__main__":
    csv_path = "../../data/raw/ActualForecastReportServlet.csv"  # Adjust this path as needed
    data = load_demand_csv(csv_path)
    data.info()
    data.head()
