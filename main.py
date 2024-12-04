from src.preprocess import Data

if __name__ == "__main__":
    data_handler = Data(r"D:\Sandu\Facultate\Anul3_sem1\ML\AP1\SEN-predictor\data\raw_data\Grafic_SEN.xlsx")
    data = data_handler.load_data()
    grouped_data = data_handler.group_data()
    aggregated_data = data_handler.aggregate_data()
    data_handler.save_grouped_data(grouped_data)
    data_handler.save_aggregated_data()