from datasets import load_dataset
import pandas as pd

# Load the dataset with specified configuration and revision
dataset_id = "neuralsorcerer/air-quality"
config = "default"
split = "train"
revision = "0b3b3f3e9408b76551f61b3b3d545a6761f2eeec"

raw_data = load_dataset(dataset_id, config, split=split, revision=revision)

# Convert to pandas for inspection
df = raw_data.to_pandas()

# Report dataset structure
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Data types:\n', df.dtypes)
print('Missing values:\n', df.isnull().sum())
print('Example rows:\n', df.head())
print('Timestamp column present:', 'timestamp' in df.columns)

# Identify CO sensor columns and environmental covariates
co_sensors = [col for col in df.columns if 'CO' in col]
env_covariates = [col for col in df.columns if col not in co_sensors and col not in ['timestamp']]

print('\nCO Sensor Columns:', co_sensors)
print('Environmental Covariates:', env_covariates)