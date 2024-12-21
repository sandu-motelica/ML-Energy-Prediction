# ML-Energy-Prediction

This project addresses a practical problem of predicting the "total balance" for Romania's National Energy System (SEN) for December 2024. The dataset describes energy consumption and production, broken down by various production sources.

## Context and Goal

The project aims to develop a solution based on **ID3 decision trees** and **Bayesian classification**, adapted for a regression problem. The main objective is to accurately predict the "total balance" for December 2024 while adhering to the following constraints:
- Data from December cannot be used for training the models.
- The solution is restricted to using **ID3** and **Bayesian** algorithms.
- Data is processed and analyzed to identify the best predictive strategies.

## Available Data

Key columns in the dataset:
- **Data**: Timestamp of the record.
- **Consum[MW]**: Total energy consumption.
- **Medie Consum[MW]**: Average consumption.
- **Producție[MW]**: Total energy production.
- **Carbune[MW]**, **Hidrocarburi[MW]**, **Ape[MW]**, **Nuclear[MW]**, **Eolian[MW]**, **Foto[MW]**, **Biomasă[MW]**: Production broken down by sources.
- **Sold[MW]**: Difference between production and consumption.

The dataset is downloaded from Transelectrica's website: **SEN Grafic**.

## Approach

### Methodology

1. **Data Preprocessing**:
   - Validation and cleaning of the dataset.
   - Adding additional attributes (e.g., time, constant/intermittent production).
   - Discretization of continuous variables (for the Bayesian algorithm).

2. **Algorithm Adaptation**:
   - **ID3**:
     - Use bucketing for the target variable `Sold[MW]`.
     - Compute feature importance for better interpretability.
   - **Bayesian Classification**:
     - Discretize continuous variables and compute conditional probabilities.

3. **Performance Evaluation**:
   - Metrics: **MAE**, **RMSE**, and **R²**.
   - Compare model performances across different strategies.

4. **Experimentation**:
   - Various approaches:
     - Direct prediction of `Sold[MW]`.
     - Predict components (`Consum[MW]` and `Producție[MW]`) and derive `Sold[MW]`.

### Workflow

1. **Data Preprocessing**:
   ```bash
   python main.py
   ```

2. **Model Training**:
   ID3 and Bayesian models are trained and compared using various feature sets.

3. **Results**:
   Results of the experiments are summarized in the console and a detailed report.

## Project Structure

```
├── data
│   ├── raw_data       # Raw data files
│   ├── processed_data # Processed data for modeling
├── reports            # Reports and documentation
├── src
│   ├── preprocess.py  # Data preprocessing logic
│   ├── id3.py         # ID3 decision tree implementation
│   ├── bayes.py       # Bayesian classifier implementation
│   ├── train.py       # Model training logic
│   ├── evaluate.py    # Evaluation metrics computation
├── main.py            # Main entry point of the project
├── README.md          # Project documentation (this file)
├── .gitignore         # Git ignore file
```

## Requirements

- **Python 3.8+**
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `openpyxl`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the raw data files in the `data/raw_data` directory.

## Results

The models are evaluated using:
- **Direct Prediction** of `Sold[MW]` using ID3.
- **Component Prediction** (`Consum[MW]` and `Producție[MW]`) to derive `Sold[MW]`.
- **Bucket Prediction** with Bayesian classification.

Performance metrics:
- **MAE**: Mean Absolute Error.
- **RMSE**: Root Mean Squared Error.
- **R²**: Coefficient of Determination.

Detailed results and comparative analysis are provided in the `reports` folder.

## Contribution

This project is developed for educational purposes.
