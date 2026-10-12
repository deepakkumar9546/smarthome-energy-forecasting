
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/household_power_consumption.txt")
OUTPUT_PATH = Path("data/processed_energy.csv")

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep=";",
    na_values=["?"],
    low_memory=False,
    usecols=["Date", "Time", "Global_active_power"]
)

print("Converting timestamps and power readings...")

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)

df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

df = df.dropna(subset=["Datetime", "Global_active_power"])
df = df.sort_values("Datetime")
df = df.set_index("Datetime")

print("Resampling to 15-minute intervals...")

# Average the valid one-minute readings in each interval.
power_15min = df["Global_active_power"].resample("15min").mean()

# Fill missing intervals using time-based interpolation.
power_15min = power_15min.interpolate(method="time", limit=4)

# Remove intervals that remain missing, including long data gaps.
power_15min = power_15min.dropna()

result = power_15min.to_frame(name="Global_active_power")
result.to_csv(OUTPUT_PATH)

print("\nPreprocessing completed!")
print(f"Cleaned readings: {len(result):,}")
print(f"Start: {result.index.min()}")
print(f"End:   {result.index.max()}")
print(f"Remaining missing values: {result.isna().sum().sum()}")
print(f"Saved to: {OUTPUT_PATH}")
print("\nFirst five processed readings:")
print(result.head())
