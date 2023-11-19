import pandas as pd

# Sample DataFrame
data = {'A': [1, 2, 3],
        'B': ['X', 'Y', 'Z'],
        'C': [4.0, 5.0, 6.0]}

df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Set column 'B' as the index
df.set_index('B', inplace=True)

# Display the DataFrame with the new index
print("\nDataFrame with 'B' as the index:")
print(df)
