import glob
import numpy as np
import random
import os
import shutil

l1 = glob.glob('Samples/candidates/**/*.bin', recursive=True)
l2 = glob.glob('Samples/annotations/**/*.bin', recursive=True)

candidates = {1: set(), 0: set()}
annotations = {1: set(), 0: set()}

def fill(container, l, label):
	while(len(container) < 100):
		file = random.choice(l)
		with open(file, 'rb') as f:
			c = np.frombuffer(f.read(8), dtype=np.int64)
			if c == label:
				print(c)
				container.add(file)


fill(candidates[1], l1, 1)
fill(candidates[0], l1, 0)
fill(annotations[1], l2, 1)

os.makedirs('Recheck/candidates/1')
os.makedirs('Recheck/candidates/0')
os.makedirs('Recheck/annotations/1')
os.makedirs('Recheck/annotations/0')

for f in candidates[1]:
	shutil.copy2(f, 'Recheck/candidates/1')
for f in candidates[0]:
	shutil.copy2(f, 'Recheck/candidates/0')
for f in annotations[1]:
	shutil.copy2(f, 'Recheck/annotations/1')
for f in annotations[0]:
	shutil.copy2(f, 'Recheck/annotations/0')
