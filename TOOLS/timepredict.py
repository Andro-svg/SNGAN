from __future__ import print_function
import time
import sys
#import datetime
#datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S');

class time_predictor():
	def __init__(self,name,max_progress):
		self.name == name
		self.max_progress = max_progress
		
		self.start_time = time.time()
		self.pred_time = self.start_time()
	
	def update(self,progress,Print = True):
		if progress > self.max_progress:
			print("Exceeded")
			return
		self.pred_time = ( ( time.time() - self.start_time )* self.max_progress / (float) progress ) + self.start_time
		print("%d / %d"%(progress,self.max_progress))
		print(time.strftime('FINISH AT:%b-%d %Y %a %H:%M:%S',self.pred_time))
