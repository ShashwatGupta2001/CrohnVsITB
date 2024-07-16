import pandas as pd
import json

# Load the JSON data from a file
with open('seg.json', 'r') as file:
    data = json.load(file)

# Convert the JSON data to a pandas DataFrame
df = pd.DataFrame(data).transpose()

# Save the DataFrame to an Excel file
file_path = "2D_json_to_table.xlsx"
df.to_excel(file_path, index=True)

print(f"The file has been saved as {file_path}")
