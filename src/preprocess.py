import pandas as pd
import os

class Data:
    def __init__(self, xlsx_path):
        self.path = xlsx_path
        self.data = None

    def load_data(self):
        self.data = pd.read_excel(self.path)
        self.data['Data'] = pd.to_datetime(self.data['Data'], dayfirst=True, errors='coerce')
        self.data = self.data.dropna(subset=['Data'])
        print("Data successfully loaded.")
        return self.data

    def group_data(self):
        grouped_data = {}
        for year in self.data['Data'].dt.year.unique():
            jan_nov = self.data[(self.data['Data'].dt.year == year) & (self.data['Data'].dt.month <= 11)]
            december = self.data[(self.data['Data'].dt.year == year) & (self.data['Data'].dt.month == 12)]
            grouped_data[f"{year}_jan_nov"] = jan_nov
            grouped_data[f"{year}_dec"] = december
        print("Data successfully grouped by year and month.")
        return grouped_data

    def aggregate_data(self, freq='h'):
        self.data['Data'] = pd.to_datetime(self.data['Data'], dayfirst=True, errors='coerce')

        self.data = self.data.dropna(subset=['Data'])

        self.data.set_index('Data', inplace=True)

        self.data = self.data.apply(pd.to_numeric, errors='coerce')

        self.data = self.data.resample(freq).sum(numeric_only=True).reset_index()

        print(f"Data successfully aggregated to {freq} level.")
        return self.data

    def save_grouped_data(self, grouped_data, output_dir="data\\processed_data"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for group_name, df in grouped_data.items():
            output_path = f"{output_dir}/{group_name}.csv"
            df.to_csv(output_path, index=False)
            print(f"File saved: {output_path}")

    def save_aggregated_data(self, output_path="data\\processed_data\\aggregated_data.csv"):
        self.data.to_csv(output_path, index=False)
        print(f"Aggregated data saved to: {output_path}")