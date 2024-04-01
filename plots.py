import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the Excel file
file_path = 'plot info.xlsx'  # Make sure to update this path to your actual file location
data = pd.read_excel(file_path, sheet_name=0)  # Assuming the data is in the first sheet

# Extracting max depth values from the first column
data['Max Depth'] = data['Unnamed: 0'].str.extract('(\d+)').astype(int)

# Plotting Memory Usage over Max Depth
plt.figure(figsize=(12, 6))

# Plot for Average Memory Usage
plt.plot(data['Max Depth'], data['Average Memory Usage (MB)'], label='Average Memory Usage', marker='o')

# Plot for Peak Memory Usage
plt.plot(data['Max Depth'], data['Peak Memory Usage (MB)'], label='Peak Memory Usage', marker='x')

plt.title('Memory Usage over Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Memory Usage (MB)')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Execution Time over Max Depth
plt.figure(figsize=(12, 6))
plt.plot(data['Max Depth'], data['Execution Time (s)'], label='Execution Time', color='red', marker='s')
plt.title('Execution Time over Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Execution Time (s)')
plt.legend()
plt.grid(True)
plt.show()
