import pandas as pd
from main import data
from framework.util import save_tsv
df=pd.DataFrame(columns=['N'])

for s in data.SETTINGS:
    df.loc[s.name]=len(s.load_data())

df.rename(index=data.IN_PAPER_NAMES, inplace=True)

print(df)
save_tsv(df, 'corpus_statistics.tsv')
