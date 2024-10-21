import pandas as pd
import arff

# Load ARFF file
with open('/home/kalyan-neeraj/Documents/ssd/bank-additional-ful-nominal.arff') as file:
    arff_data = arff.load(file)

# Convert to DataFrame
df = pd.DataFrame(arff_data['data'], columns=[attribute[0] for attribute in arff_data['attributes']])

# Save DataFrame to CSV
df.to_csv('bank.csv', index=False)