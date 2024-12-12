import os
from dotenv import load_dotenv
import pandas as pd

### This script is used to extract the corresponding labels for each CT Scan ###


# Load variables from the .env file
load_dotenv()
output_path = os.getenv("output_path")
csv_path = os.getenv("csv_path")

# Read in csv containing target labels of training data
df = pd.read_csv(csv_path)
dictionary = df.set_index('Key')['Value'].to_dict()
img_list = os.listdir(output_path)

labels=[]
actual_values = []

### Matches each Image-ID with its respective labels in the training_data_labels.csv ###
### Only extracts the _any label (is there a hemorrhage or not) ####

for file_name in img_list:
    # Remove the .png extension (or any other extension)
    key = os.path.splitext(file_name)[0]

    if key in dictionary:
        # Get the last label from the list for the current file
        label = int(dictionary[key][-2])
        true_value = int(df[df['Key'] == key]['Value'].iloc[0][-2])

        # Append the label to both lists
        labels.append(label)
        actual_values.append(true_value)
    else:
        # Handle the case where the key is not found in the grouped data
        print(f"Warning: {key} not found in grouped_data")


print("Start Cross-Checking if the labels correspond correctly")
# Assert that extracted labels ("labels") match original labels ("actual_labels")
if labels == actual_values:
    print("The labels match this_list!")
else:
    raise AssertionError("The labels do not match this_list!")


### Optional: Export Labels ##
# Convert the 'labels' list into a DataFrame

# labels_df = pd.DataFrame(labels, columns=['Label'])
# output_csv_path = "../testdatalabels.csv"
# labels_df.to_csv(output_csv_path, index=False)
# print(f"Labels have been successfully exported to {output_csv_path}")
