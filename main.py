import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import Data
from src.train import train_and_evaluate_models

if __name__ == "__main__":
    data_handler = Data(r"D:\Sandu\Facultate\Anul3_sem1\ML\AP1\ML-Energy-Prediction\data\raw_data\Grafic_SEN.xlsx")
    data = data_handler.load_data()
    # grouped_data = data_handler.group_data()
    # aggregated_data = data_handler.aggregate_data()
    # data_handler.save_grouped_data(grouped_data)
    # data_handler.save_aggregated_data()

    data_path = "data/processed_data/aggregated_data.csv"
    features = ["Consum[MW]", "Productie[MW]", "Carbune[MW]", "Hidrocarburi[MW]", "Ape[MW]",
                "Nuclear[MW]", "Eolian[MW]", "Foto[MW]", "Biomasa[MW]"]
    target = "Sold[MW]"

    results = train_and_evaluate_models(data_path, features, target)

    print("Final Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")