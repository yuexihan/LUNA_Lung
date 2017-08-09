import os
import re
import json

d = json.load(open('subset.json'))

for subset in d:
	folder = os.path.join('LUNA_data', subset)
	os.mkdir(folder)
	for f in d[subset]:
		origin = os.path.join('LUNA_data', f)
		target = os.path.join(folder, f)
		os.rename(origin, target)