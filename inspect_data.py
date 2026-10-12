
import pandas as pd
from pathlib import Path

# Path to the downloaded dataset
DATA_PATH = Path("data/household_power_consumption.txt")

print("Loading household electricity dataset...")

# Read the semicolon-separated file
df = pd.read_csv(
    DATA_PATH,
    sep=";",
    na_values=["?"],
    low_memory=False
)

print("\nDataset loaded successfully!")
print("-" * 50)

print(f"Number of rows: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# Combine date and time into one timestamp
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)

# Convert active power to numeric
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

print("\nDate range:")
print("Start:", df["Datetime"].min())
print("End:  ", df["Datetime"].max())

print("\nMissing values:")
print(df[["Datetime", "Global_active_power"]].isna().sum())

print("\nAverage active power (kW):")
print(round(df["Global_active_power"].mean(), 3))

print("\nDataset inspection completed!")
