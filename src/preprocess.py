import os
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


class Data:
    def __init__(self, excel_paths = None):
        self.excel_paths = excel_paths

    def process_and_save_csv(self, destination_folder="data\\processed_data", add_time_attributes=False,
                             add_production_attributes=False):
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        all_data = []
        december_data = []

        for path in self.excel_paths:
            year = int(os.path.basename(path).split('.')[0])
            data = pd.read_excel(path)

            data = self.remove_invalid_rows(data)

            data['Data'] = pd.to_datetime(data['Data'], dayfirst=True, errors='coerce')
            data = data.dropna(subset=['Data'])

            self.validate_data(data)

            if add_time_attributes:
                data = self.add_time_attributes(data)

            if add_production_attributes:
                data = self.add_production_attributes(data)

            csv_path = os.path.join(destination_folder, f"{year}.csv")
            data.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

            december = data[data['Data'].dt.month == 12]
            december_data.append(december)

            data = data[data['Data'].dt.month < 12]
            all_data.append(data)

        concatenated_data = pd.concat(all_data, ignore_index=True)
        concatenated_path = os.path.join(destination_folder, "all_years.csv")
        concatenated_data.to_csv(concatenated_path, index=False)
        print(f"Saved all data (excluding December): {concatenated_path}")

        december_data = pd.concat(december_data, ignore_index=True)
        december_path = os.path.join(destination_folder, "all_december.csv")
        december_data.to_csv(december_path, index=False)
        print(f"Saved all December data: {december_path}")

    @staticmethod
    def validate_data(data):
        required_columns = ['Eolian[MW]', 'Foto[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]',
                            'Ape[MW]', 'Nuclear[MW]', 'Data']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

    @staticmethod
    def add_time_attributes(data):
        data['Year'] = data['Data'].dt.year
        data['Month'] = data['Data'].dt.month
        data['Day_of_Week'] = data['Data'].dt.dayofweek
        data['Hour'] = data['Data'].dt.hour

        return data

    @staticmethod
    def add_production_attributes(data):
        data['Intermittent_Production'] = data['Eolian[MW]'] + data['Foto[MW]']
        data['Constant_Production'] = (data['Carbune[MW]'] + data['Hidrocarburi[MW]'] +
                data['Ape[MW]'] + data['Nuclear[MW]'])

        return data

    @staticmethod
    def remove_invalid_rows(data):
        invalid_rows = data.apply(lambda row: any('*' in str(value) for value in row), axis=1)

        cleaned_data = data[~invalid_rows]
        return cleaned_data

    @staticmethod
    def load_csv(csv_path):
        try:
            data = pd.read_csv(csv_path, low_memory=False)

            data['Data'] = pd.to_datetime(
                data['Data'],
                format="%Y-%m-%d %H:%M:%S",
                errors='coerce'
            )
            data = data.dropna(subset=['Data'])
            print(f"Data loaded from {csv_path}.")
            return data

        except Exception as e:
            print(f"Error: {e}")
            raise

    @staticmethod
    def get_train_test_split(data, year, test_month=12):
        if data is None:
            raise ValueError("Data is not loaded. Use load_csv().")

        train_data = data[(data['Year'] == year) & (data['Data'].dt.month < test_month)]
        test_data = data[(data['Year'] == year) & (data['Data'].dt.month== test_month)]
        return train_data, test_data

    @staticmethod
    def discretize_columns(data, columns, n_bins=10, encode='ordinal', strategy='uniform'):
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
        data[columns] = discretizer.fit_transform(data[columns])
