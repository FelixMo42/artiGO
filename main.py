#imports

import tensorflow as tf
import numpy as np
import math

#creat simulation

map = np.zeros((1000,1000), dtype=int)

def draw(sx,sy,w,h):
	map[sx : sx + w,sy : sy + h] = 1

bot = {
	"pos" : [0, 0],
	"size" : [20, 60]
}

width = 1000
height = 1000

target = [900, 900]

draw(500 - 30,500 - 30,60,60)

def raycast(sx,sy,a):
	if a == 180:
		x = sx
		for y in range(sy, 0, -1):
			if map[x][y] == 1:
				return (x, y)
	elif a > 0:
		for x in range(0, width - sx):
			y = sx + math.floor(x * 1)
			if map[x][y] == 1:
				return (x, y)
	elif a < 0:
		for x in range(0, sx):
			y = sx - math.floor(x * 1)
			if map[x][y] == 1:
				return (x, y)
	elif a == 0:
		x = sx
		for y in range(sy, height):
			if map[x][y] == 1:
				return (x, y)

print( raycast(0,0,12) )
