import pandas as pd
import arff

path_to_arff = 'bank-full.arff'
# Load ARFF file
with open(path_to_arff) as file:
    arff_data = arff.load(file)

# Convert to DataFrame
df = pd.DataFrame(arff_data['data'], columns=[attribute[0] for attribute in arff_data['attributes']])

# Save DataFrame to CSV
df.to_csv('bank.csv', index=False)