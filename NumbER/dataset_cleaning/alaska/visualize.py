import json
import matplotlib.pyplot as plt
from collections import Counter

# Load the array of dictionaries from the file
with open('all_data.json', 'r') as f:
    data = json.load(f)

print(len(data))
# Flatten the dictionaries into a list of keys
keys = [key for dict_ in data for key in dict_.keys()]
# Count the occurrence of each key
counts = dict(Counter(keys))

# Calculate the percentage of dictionaries each key appears in
percentages = {key: count / len(data) * 100 for key, count in counts.items()}

# Create a bar chart of the percentages
fig, ax = plt.subplots()
ax.bar(percentages.keys(), percentages.values())
ax.set_xlabel('Keys')
ax.set_ylabel('Percentage of Dictionaries')
ax.set_title('Percentage of Dictionaries Each Key Appears In')
plt.show()