import pandas as pd
import re

def clean_pattern(x, pattern):
    x = str(x)
    match = re.search(pattern, x)
    if match:
        weight = match.group(1)# or match.group(3)
        weight_value = float(weight)
        return weight_value
    else:
        return ''

df = pd.read_csv('pat.h')
    
#clean weight
kg_pattern = r'(\d+(?:\.\d+)?)\s*(k|K)g'
g_pattern = r'(\d+(?:\.\d+)?)\s*g\b'
lb_pattern = r'(\d+(?:\.\d+)?)\s*(lb|lbs|LBS|pounds|Pounds|Pound|pound)\b'
oz_pattern = r'(\d+(?:\.\d+)?)\s*oz\b'
df['weight_lb'] = df['weight'].apply(clean_pattern, args=(lb_pattern,))
df['weight_oz'] = df['weight'].apply(clean_pattern, args=(oz_pattern,))
df['weight_kg'] = df['weight'].apply(clean_pattern, args=(kg_pattern,))
df['weight_g'] = df['weight'].apply(clean_pattern, args=(g_pattern,))


#clean dimensions   

#df[['weight', 'weight_lb', 'weight_kg', 'weight_oz', 'weight_g']].to_csv('temp.csv', index=False)
    