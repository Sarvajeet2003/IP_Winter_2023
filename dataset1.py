import pandas as pd

# Specify the path to your text file

file_path = 'C:\\Users\\ayush\\OneDrive\\Desktop\\IP-sem6\\sampe+.txt'

# Rest of your code...



# Read content from the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Create lists to store data
ids = []
sequences = []

# Process each line in the file
current_id = ""
for line in lines:
    if line.startswith('>'):
        # If the line starts with '>', it is an ID line
        current_id = line.strip()
    else:
        # If the line does not start with '>', it is a sequence line
        ids.append(current_id)
        sequences.append(line.strip())

# Create a DataFrame
df = pd.DataFrame({'ID': ids, 'Sequence': sequences})

# Strip leading and trailing whitespaces from the 'ID' column
df['ID'] = df['ID'].str.strip()

# Display the DataFrame
print(df)


# Read content from the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Create lists to store data
ids = []
sequences = []

# Process each line in the file
current_id = ""
for line in lines:
    if line.startswith('>'):
        # If the line starts with '>', it is an ID line
        current_id = line.strip()
    else:
        # If the line does not start with '>', it is a sequence line
        ids.append(current_id)
        sequences.append(line.strip())

# Create a DataFrame
df = pd.DataFrame({'ID': ids, 'Sequence': sequences})

# Strip leading and trailing whitespaces from the 'ID' column
df['ID'] = df['ID'].str.strip()

# Display the DataFrame
print(df)















# file_path = 'C:\\Users\\ayush\\OneDrive\\Desktop\\IP-sem6\\sampe+.txt'

# # Rest of your code...

# # Read content from the file
# with open(file_path, 'r') as file:
#     lines = file.readlines()

# # Add the identifier to the beginning of each sequence
# modified_lines = []
# current_identifier = ""

# for line in lines:
#     if line.startswith('>'):
#         current_identifier = line.strip()
#     else:
#         modified_lines.append(current_identifier + ' : ' + line.strip())

# # Write the modified content back to the file
# with open(file_path, 'w') as file:
#     file.write('\n'.join(modified_lines))
