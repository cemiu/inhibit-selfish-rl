import pandas as pd

# create example dataframe
df = pd.DataFrame({'run': ['tom_XTJD', 'tom_YLSA', 'max_SKTA', 'max_SJFL'],
                   'speed': [19, 22, 22, 17],
                   'distance': [130, 160, 156, 110]})

# extract runner names from the run column
df['runner'] = df['run'].str.split('_', expand=True)[0]

grouped = df.groupby('runner').agg({'speed': ['mean', 'std'], 'distance': ['mean', 'std']})
# grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

# display results
print(grouped)
