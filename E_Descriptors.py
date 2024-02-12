
import csv
import numpy as np

# Function to calculate autocovariance
def autocovariance(sequence, lag):
    n = len(sequence)
    mean = np.mean([ord(aa) for aa in sequence])
    result = 0
    for i in range(n - lag):
        result += (ord(sequence[i]) - mean) * (ord(sequence[i + lag]) - mean)
    return result / n

# Read sequences from CSV file and calculate mean E-descriptors
input_csv_path = 'wk2_ip_datasets/training_neg.csv' 
output_csv_path = 'training_neg_E_Descriptors.csv.csv'  
sequences = []

with open(input_csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader, None)
    sequences_col_index = header.index('Sequence') 
    sequences = [row[sequences_col_index] for row in csv_reader]

# Calculate mean E-descriptors for each sequence
lags = [1, 2, 3, 4, 5]
E_descriptors_list = []

for sequence in sequences:
    E_descriptors = []
    for lag in lags:
        autocovariance_values = [autocovariance(sequence, lag) for _ in range(5)]
        mean_autocovariance = np.mean(autocovariance_values)
        E_descriptors.append(mean_autocovariance)
    E_descriptors_list.append(E_descriptors)

with open(output_csv_path, 'w', newline='') as csv_output_file:
    csv_writer = csv.writer(csv_output_file)
    header_row = ['Sequence'] + [f'Mean_E{j + 1}' for j in range(5)]
    csv_writer.writerow(header_row)
    for i, sequence in enumerate(sequences):
        row_data = [sequence] + E_descriptors_list[i]
        csv_writer.writerow(row_data)
print(f"Mean E-descriptors have been written to {output_csv_path}")
