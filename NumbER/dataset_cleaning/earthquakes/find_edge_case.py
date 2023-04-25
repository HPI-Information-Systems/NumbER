# Edge case exists when we have three earthquakes. One in catalog A (1) and two in catalog B (2,3).
# It could now happen that earthquake 2 is within the 16 seconds to 1 and 3 is within the 16 seconds to 2. 
# This could be handled in two ways: First, saying that all earthquakes represent the same event or second, 
# saying that only one of the two earthquakes matches 1, but how is selected, which earthquake is selected?
import pandas as pd

df = pd.read_csv('temp.csv', parse_dates=['time'])

utah = df[df['datasource'] == 'uu']
#utah.to_csv('./utah.csv', index=False)
us = df[df['datasource'] == 'us']
final_result = pd.DataFrame()
for idx, earthquake in utah.iterrows():
    earthquake.to_csv('./earthquake.csv')
    candidates = us[(us['time']-earthquake['time']).between(pd.Timedelta(seconds=-16), pd.Timedelta(seconds=16))==True]
    candidates.to_csv('./candidates.csv', index=False)
    if len(candidates)> 1:
        final_result = final_result.append(earthquake)
        final_result = final_result.append(candidates)
        final_result = final_result.append(pd.Series(), ignore_index=True)
        
final_result.to_csv('./final_result.csv', index=False)

