import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

from datetime import datetime
import pandas as pd
import pandas_datareader.data as web

# replaces pyfinance.ols.PandasRollingOLS (no longer maintained)
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
idx = pd.IndexSlice

DATA_STORE = '../data/assets.h5'

START = 2000
END = 2018

with pd.HDFStore(DATA_STORE) as store:
    prices = (store['quandl/wiki/prices']
              .loc[idx[str(START):str(END), :], 'adj_close']
              .unstack('ticker'))
    stocks = store['us_equities/stocks'].loc[:, ['marketcap', 'ipoyear', 'sector']]

prices.info()

stocks.info()

stocks = stocks[~stocks.index.duplicated()]
stocks.index.name = 'ticker'

shared = prices.columns.intersection(stocks.index)

stocks = stocks.loc[shared, :]
stocks.info()

prices = prices.loc[:, shared]
prices.info()

assert prices.shape[1] == stocks.shape[0]

monthly_prices = prices.resample('M').last()

monthly_prices.info()

outlier_cutoff = 0.01
data = pd.DataFrame()
lags = [1, 2, 3, 6, 9, 12]
for lag in lags:
    data[f'return_{lag}m'] = (monthly_prices
                           .pct_change(lag)
                           .stack()
                           .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                  upper=x.quantile(1-outlier_cutoff)))
                           .add(1)
                           .pow(1/lag)
                           .sub(1))
data = data.swaplevel().dropna()
data.info()

min_obs = 120
nobs = data.groupby(level='ticker').size()
keep = nobs[nobs > min_obs].index

data = data.loc[idx[keep,:], :]
data.info()

data.describe()

# cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.clustermap(data.corr('spearman'), annot=True, center=0, cmap='Blues');



factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2000')[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'
factor_data.info()

factor_data = factor_data.join(data['return_1m']).sort_index()
factor_data.info()

T = 24
betas = (factor_data.groupby(level='ticker', group_keys=False)
         .apply(lambda x: RollingOLS(endog=x.return_1m,
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(T, x.shape[0]-1))
                .fit(params_only=True)
                .params
                .drop('const', axis=1)))

betas.describe().join(betas.sum(1).describe().to_frame('total'))

betas.describe().join(betas.sum(1).describe().to_frame('total'))

cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.clustermap(betas.corr(), annot=True, cmap=cmap, center=0);

data = data.join(betas.groupby(level='ticker').shift())
data.info()

data[factors] = data.groupby('ticker')[factors].transform(lambda x: x.fillna(x.mean()))
data.info()

print(factors)  # Check if 'factors' is valid
print(data.columns)  # List all available columns




# Ensure 'factors' are the valid column names in your dataframe


for lag in [2,3,6,9,12]:
    data[f'momentum_{lag}'] = data[f'return_{lag}m'].sub(data.return_1m)
data[f'momentum_3_12'] = data[f'return_12m'].sub(data.return_3m)

dates = data.index.get_level_values('date')
data['year'] = dates.year
data['month'] = dates.month



"""dates = data.index.get_level_values('date')
data['year'] = dates.year
data['month'] = dates.month
"""

# Inspect the current structure of the DataFrame
print(data.columns)  # Check the columns in the DataFrame
print(data.index.names)  # Check the names of the index levels


# Shift 'return_1m' based on the 'ticker' level of the index
for t in range(1, 7):
    data[f'return_1m_t-{t}'] = data.groupby(level='ticker')['return_1m'].shift(t)

# Display information about the DataFrame
data.info()


for t in [1,2,3,6,12]:
    data[f'target_{t}m'] = data.groupby(level='ticker')[f'return_{t}m'].shift(-t)

cols = ['target_1m',
        'target_2m',
        'target_3m', 
        'return_1m',
        'return_2m',
        'return_3m',
        'return_1m_t-1',
        'return_1m_t-2',
        'return_1m_t-3']

data[cols].dropna().sort_index().head(10)

data.info()

data = (data
        .join(pd.qcut(stocks.ipoyear, q=5, labels=list(range(1, 6)))
              .astype(float)
              .fillna(0)
              .astype(int)
              .to_frame('age')))
data.age = data.age.fillna(-1)

stocks.info()



print(data.columns)


print(data.index.names)  # Check the names of index levels
print(data.columns)  # Check the columns in the DataFrame


print(monthly_prices.index.names)  # Check the names of the index levels
print(monthly_prices.head())       # Inspect the first few rows of the DataFrame


# Sort by 'date' index descending and calculate pct_change across all tickers (columns)
size_factor = (monthly_prices
               .sort_index(ascending=False)  # Sort by 'date' index descending
               .pct_change()  # Calculate percentage change for each ticker
               .fillna(0)  # Fill missing values with 0
               .add(1)  # Add 1 to pct_change to get cumulative product
               .cumprod())  # Get cumulative product for returns

# Display information about the 'size_factor' DataFrame
size_factor.info()

# Preview the first few rows of the result
print(size_factor.head())


msize = (size_factor
         .mul(stocks
              .loc[size_factor.columns, 'marketcap'])).dropna(axis=1, how='all')

data['msize'] = (msize
                 .apply(lambda x: pd.qcut(x, q=10, labels=list(range(1, 11)))
                        .astype(int), axis=1)
                 .stack()
                 .swaplevel())
data.msize = data.msize.fillna(-1)

data = data.join(stocks[['sector']])
data.sector = data.sector.fillna('Unknown')

data.info()

# Check the index structure
print(data.index.names)
print(data.head())


with pd.HDFStore(DATA_STORE) as store:
    store.put('engineered_features', data.sort_index().loc[idx[:, :datetime(2018, 3, 1)], :])
    print(store.info())

dummy_data = pd.get_dummies(data,
                            columns=['year','month', 'msize', 'age',  'sector'],
                            prefix=['year','month', 'msize', 'age', ''],
                            prefix_sep=['_', '_', '_', '_', ''])
dummy_data = dummy_data.rename(columns={c:c.replace('.0', '') for c in dummy_data.columns})
dummy_data.info()

import nbformat

# Path to your current notebook (you can hardcode the file name)
notebook_filename = '01_feature_engineering.ipynb'

# Open the notebook
with open(notebook_filename) as f:
    nb = nbformat.read(f, as_version=4)

# Extract all code cells
code_cells = [cell['source'] for cell in nb['cells'] if cell['cell_type'] == 'code']

# Combine and print all code cells
all_code = '\n\n'.join(code_cells)

# Print all code (or you can save it to a file if you prefer)
print(all_code)

# Optionally, write the code to a file
with open("all_code.py", "w") as f:
    f.write(all_code)




