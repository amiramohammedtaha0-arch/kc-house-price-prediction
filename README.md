# King County House Price Prediction

A regression project that predicts house sale prices in King County, WA, using a public housing dataset. The project includes full data cleaning/feature engineering, a **linear regression model implemented from scratch** (mini-batch gradient descent with NumPy), and a comparison against scikit-learn implementations (with polynomial features and regularization).

## Why this project

I wanted to understand linear regression at the mechanical level — not just call `.fit()` — so I implemented gradient descent manually and then compared it against scikit-learn's optimized implementations to validate correctness and see how far feature engineering (polynomial features, regularization) could push performance.

## Dataset

`kc_house_data.csv` — King County house sales data, containing features such as bedrooms, bathrooms, square footage, floors, waterfront/view flags, condition, year built/renovated, and location.

## Data Preprocessing (`preperaion.py`)

- Dropped `id` and `zipcode` (identifiers, not directly useful without further encoding).
- Extracted `sale_year` from the sale date and derived `age = sale_year - yr_built`.
- Dropped `sqft_above` due to strong correlation with `sqft_living` (multicollinearity).
- Dropped low-correlation columns: `sale_year`, `sqft_lot`, `sqft_lot15`, `condition`, `date`.
- Removed clear outliers (e.g. `bathrooms == 0` with `sqft_living > 1000`, `bedrooms > 15`).
- Applied `log1p` to skewed numeric features: `sqft_living`, `sqft_living15`, `price`.
- Converted `waterfront`, `yr_renovated`, and basement presence into binary flags.

## Modeling

1. **From scratch** (`modelImp.py`) — Linear regression trained with mini-batch gradient descent implemented directly in NumPy, including standardization and a convergence check based on gradient norm and cost change between epochs.
2. **scikit-learn baseline** (`sklearnLR.py`) — `LinearRegression` on degree-3 polynomial features with standard scaling.
3. **scikit-learn pipeline** (`sklearnPipeline.py`) — `RobustScaler` → `PolynomialFeatures(degree=3)` → `Ridge` regression, wrapped in a single `Pipeline`.


## Results

| Model | R² |
|---|---|
| From-scratch gradient descent | 0.756 *(on full dataset, not train/test split — see note below)* |
| scikit-learn LinearRegression + poly features | valid: 0.817, test: 0.821 |
| scikit-learn Pipeline (RobustScaler + poly + Ridge) | train: 0.856, valid: 0.849, test: 0.833 |

> **Note:** The from-scratch model's R² was computed on the full dataset (no train/test split), so it isn't directly comparable to the sklearn models' validation/test scores above. A fair comparison would require splitting the data before training the from-scratch model — this is a known limitation and a natural next step.

*(Run the scripts below and fill in the printed R² values.)*

## Project Structure

```
kc-house-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── kc_house_data.csv
├── notebooks/
│   └── preperaion.ipynb
└── src/
    ├── preperaion.py
    ├── modelImp.py
    ├── sklearnLR.py
    └── sklearnPipeline.py
```

## How to Run

```bash
pip install -r requirements.txt

# 1. Clean the raw data and generate dataAfterPre.csv
python src/preperaion.py

# 2. Train the from-scratch gradient descent model
python src/modelImp.py

# 3. Compare against scikit-learn models
python src/sklearnLR.py
python src/sklearnPipeline.py
```

## Possible Next Steps

- Cross-validation instead of a single train/valid/test split.
- Hyperparameter tuning for the Ridge regularization strength.
- Feature encoding for `zipcode` (e.g. target encoding) instead of dropping it.
