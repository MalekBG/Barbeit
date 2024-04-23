import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the Excel file
file_path = 'plot info.xlsx'  # Make sure to update this path to your actual file location
data = pd.read_excel(file_path, sheet_name=0)  # Assuming the data is in the first sheet

# Extracting max depth values from the first column
data['Max Depth'] = data['Unnamed: 0'].str.extract('(\d+)').astype(int)

# Plotting State Number over Maximum Depth
plt.figure(figsize=(12, 6))

# Plot for Average State Number
plt.plot(data['Max Depth'], data['Average State Number'], label='Average Number of States', marker='o')

# Plot for Peak State Number
plt.plot(data['Max Depth'], data['Peak State Number'], label='Peak Number of States', marker='x')

plt.title('Number of States over Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Number of States')
plt.legend()
plt.grid(True)
plt.show()
