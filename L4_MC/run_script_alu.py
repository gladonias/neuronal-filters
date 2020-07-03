import os
import numpy as np

for i in np.arange(10, 301, 10):
	for j in range(1):
		os.system('python3 alu.py {}'.format(i))
os.system('python3 email_notification.py')