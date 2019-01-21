import os
import numpy as np
#import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from pylab import  show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes


RESULTS_FOLDER = './results/'
NUM_BINS = 1000
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
K_IN_M = 1000.0
SCHEMES = ['rl']

def main():
	time_ms = []
	packet_size = []
	delay = []
	reward = [] 
	age0 = []
	age1 = []
	age2 = []
	age3 = []
	age4 = []
	age5 = []
	age6 = []
	age7 = []
	age8 = []
	age9 = []

	
	base1_time_ms = []
	base1_packet_size = []
	base1_delay = []
	base1_reward = [] 
	base1_age0 = []
	base1_age1 = []
	base1_age2 = []
	base1_age3 = []
	base1_age4 = []
	base1_age5 = []
	base1_age6 = []
	base1_age7 = []
	base1_age8 = []
	base1_age9 = []


	base2_time_ms = []
	base2_packet_size = []
	base2_delay = []
	base2_reward = [] 
	base2_age0 = []
	base2_age1 = []
	base2_age2 = []
	base2_age3 = []
	base2_age4 = []
	base2_age5 = []
	base2_age6 = []
	base2_age7 = []
	base2_age8 = []
	base2_age9 = []
		
	proposed = []
	base_1 = []
	base_2 = []

	log_files = os.listdir(RESULTS_FOLDER)
	for log_file in log_files:
		
		print log_file

		with open(RESULTS_FOLDER + log_file, 'rb') as f:
			
			if 'base1' in log_file:
				for line in f:
					
					parse = line.split()
					if len(parse) <= 1:
						break
					base1_time_ms.append(float(parse[0]))
					base1_packet_size.append(int(parse[1]))
					base1_delay.append(float(parse[2]))
					base1_reward.append(str(parse[3])) 
					base1_age0.append(float(parse[4]))
					base1_age1.append(float(parse[5]))
					base1_age2.append(float(parse[6]))
					base1_age3.append(float(parse[7]))
					base1_age4.append(float(parse[8]))
					base1_age5.append(float(parse[9]))
					base1_age6.append(float(parse[10]))
					base1_age7.append(float(parse[11]))
					base1_age8.append(float(parse[12]))
					base1_age9.append(float(parse[13]))

			elif 'base2' in log_file:
				for line in f:
					
					parse = line.split()
					if len(parse) <= 1:
						break
					base2_time_ms.append(float(parse[0]))
					base2_packet_size.append(int(parse[1]))
					base2_delay.append(float(parse[2]))
					base2_reward.append(str(parse[3])) 
					base2_age0.append(float(parse[4]))
					base2_age1.append(float(parse[5]))
					base2_age2.append(float(parse[6]))
					base2_age3.append(float(parse[7]))
					base2_age4.append(float(parse[8]))
					base2_age5.append(float(parse[9]))
					base2_age6.append(float(parse[10]))
					base2_age7.append(float(parse[11]))
					base2_age8.append(float(parse[12]))
					base2_age9.append(float(parse[13]))

			else:
				for line in f:
					
					parse = line.split()
					if len(parse) <= 1:
						break
					time_ms.append(float(parse[0]))
					packet_size.append(int(parse[1]))
					delay.append(float(parse[2]))
					reward.append(str(parse[3])) 
					age0.append(float(parse[4]))
					age1.append(float(parse[5]))
					age2.append(float(parse[6]))
					age3.append(float(parse[7]))
					age4.append(float(parse[8]))
					age5.append(float(parse[9]))
					age6.append(float(parse[10]))
					age7.append(float(parse[11]))
					age8.append(float(parse[12]))
					age9.append(float(parse[13]))
   


	# ---- ---- ---- ----
	# CDF 
	# ---- ---- ---- ----
	j = 0
	toPlot = np.zeros((1000,1))
	toPlot_base1 = np.zeros((1000,1))
	toPlot_base2 = np.zeros((1000,1))
	fig = plt.figure()
	ax = fig.add_subplot(nrows=2, ncols=5)

	for x in range(0,10):

		ax = plt.subplot(2, 5, x+1)
		proposed = eval('age%d'% (x))
		base_1 = eval('base1_age%d'% (x))
		base_2 = eval('base2_age%d'% (x))

		values0, base0 = np.histogram(proposed, bins=NUM_BINS)
		values1, base1 = np.histogram(base_2, bins=NUM_BINS)
		values2, base2 = np.histogram(base_1, bins=NUM_BINS)
		
		cumulative0 = np.cumsum(values0)
		cumulative1 = np.cumsum(values1)
		cumulative2 = np.cumsum(values2)

		for i in range(0,len(cumulative0)):
			toPlot[i]=float(cumulative0[i])
			toPlot[i]=toPlot[i]/30000
			toPlot_base1[i]=float(cumulative1[i])
			toPlot_base1[i]=toPlot_base1[i]/30000
			toPlot_base2[i]=float(cumulative2[i])
			toPlot_base2[i]=toPlot_base2[i]/30000

		label_proposed = 'Sensor ' + str(i+1) + ' (Proposed)'
		label_base1 = 'Sensor ' + str(i+1) + ' (Basline max age)'
		label_base2 = 'Sensor ' + str(i+1) + ' (Baseline random)'
		
		titles = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']

		labels = 'Sensor ' + str(x+1)
		if x < 5:
		    A = 0
		    B = 110
		else:
		    A = 0
		    B = 220

		ax.plot(base0[:-1], toPlot, label='Proposed: ' + labels )
		ax.legend()
		ax.set_xlim((A,B))
		ax.plot(base1[:-1], toPlot_base1, label='Baseline-1: ' + labels )	
		ax.legend()
		ax.set_xlim((A,B))
		ax.plot(base2[:-1], toPlot_base2, label='Baseline-2: ' + labels )	
		ax.legend()
		ax.set_xlim((A,B))
		plt.ylabel('CDF')
		plt.xlabel('AGE \n' +titles[x])

	show()
if __name__ == '__main__':
	main()
