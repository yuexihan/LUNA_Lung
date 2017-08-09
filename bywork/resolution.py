import glob
import pandas as pd
from tqdm import tqdm
import re

mhds = glob.glob('LUNA_data/*/*.mhd')
spacings = []
for mhd in tqdm(mhds):
	s = open(mhd).read()
	spacing = re.search(r'ElementSpacing = ([.0-9]+) ([.0-9]+) ([.0-9]+)\n', s).groups()
	spacing = tuple(float(x) for x in spacing)
	spacings.append((mhd,) + spacing)

df = pd.DataFrame.from_records(spacings, columns=['seriesuid', 'X', 'Y', 'Z'])
df.to_csv('resolution.csv', index=False)