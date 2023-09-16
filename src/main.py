import pandas as pd
from data.fetch_demand_data import load_demand_csv
from database.database_operations import save_to_sqlite

def main():
    # 定义文件和数据库的路径
    csv_path = "../data/raw/ActualForecastReportServlet.csv"
    db_path = "./data/database.sqlite"  # 你可以按需更改这个路径和文件名
    table_name = "demand_data"  # 数据库中的表名，你可以按需更改

    # 1. 从CSV文件中加载数据
    data = load_demand_csv(csv_path)
    print("Loaded CSV data.")
    print(data.head())  # 打印预览数据

    # 2. 将数据保存到SQLite数据库
    save_to_sqlite(data, db_path, table_name)
    print(f"Data saved to SQLite database at {db_path} in table {table_name}.")

if __name__ == "__main__":
    main()
