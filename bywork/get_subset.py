import os
import re
import json

subset = re.compile(r'^subset[0-9]$')

d = {}
fs = os.listdir()
for f in fs:
	if subset.match(f):
		d[f] = os.listdir(f)

with open('subset.json', 'w') as f:
	json.dump(d, f)