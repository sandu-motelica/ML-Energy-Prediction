import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import Data
from src.train import *

def preprocess_data():
    excel_folder = os.path.join("data", "raw_data")
    excel_files = [
        # os.path.join(excel_folder, "2021.xlsx"),
        os.path.join(excel_folder, "2022.xlsx"),
        os.path.join(excel_folder, "2023.xlsx"),
        os.path.join(excel_folder, "2024.xlsx")
    ]
    data_handler = Data(excel_files)
    data_handler.process_and_save_csv(add_time_attributes=True, add_production_attributes=True)


def main():
    # preprocess_data()

    print("\nTest 2023 only")
    data_2023 = Data.load_csv("data/processed_data/2023.csv")
    data_train, data_test = Data.get_train_test_split(data_2023, 2023)

    feature_columns = ['Month', 'Day_of_Week', 'Hour', 'Intermittent_Production', 'Constant_Production']
    target_column = 'Sold[MW]'
    print(data_train.shape, data_test.shape)


    direct_model = direct_prediction(data_train, data_test, feature_columns, target_column)
    consum_model, productie_model = component_prediction(data_train, data_test, feature_columns)
    bucket_model = bucket_prediction(data_train, data_test, feature_columns)

    print("=== Feature Importances ===")
    print("Feature importances (Approach 1):", direct_model.feature_importances())
    print("Feature importances (Approach 2 - Consum):", consum_model.feature_importances())
    print("Feature importances (Approach 2 - Productie):", productie_model.feature_importances())
    print("Feature importances (Approach 3 - Bucketing):", bucket_model.feature_importances())

    print("\nTest all data")

    data_all = Data.load_csv("data/processed_data/all_years.csv")
    data_all_december = Data.load_csv("data/processed_data/all_december.csv")
    _, data_2023_december = Data.get_train_test_split(data_all_december, 2023)

    feature_columns = ['Month', 'Day_of_Week', 'Hour', 'Intermittent_Production', 'Constant_Production']
    target_column = 'Sold[MW]'
    print(data_all.shape, data_2023_december.shape)

    direct_model = direct_prediction(data_all, data_2023_december, feature_columns, target_column)
    consum_model, productie_model = component_prediction(data_all, data_2023_december, feature_columns)
    bucket_model = bucket_prediction(data_all, data_2023_december, feature_columns)

    print("=== Feature Importances ===")
    print("Feature importances (Approach 1):", direct_model.feature_importances())
    print("Feature importances (Approach 2 - Consum):", consum_model.feature_importances())
    print("Feature importances (Approach 2 - Productie):", productie_model.feature_importances())
    print("Feature importances (Approach 3 - Bucketing):", bucket_model.feature_importances())



if __name__ == "__main__":
    main()