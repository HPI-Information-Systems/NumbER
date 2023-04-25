import pandas as pd
from column import Column

for col in df.itercols():
    p_c = calculate_p_c(col)
    if p_c < 0.05:
        #column will not be considered as comparison column
        continue
    

