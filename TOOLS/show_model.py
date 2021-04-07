import torch
import sys

if __name__ == '__main__':
	D = torch.load(sys.argv[1])
	for key in D:
		print(key)
