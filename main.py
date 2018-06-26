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
	"size" : [20, 60],
	"angle" : 0
}

width = 1000
height = 1000

target = [900, 900]

draw(500 - 30,500 - 30,60,60)

def raycast(sx,sy,a):
	if a == 180:
		x = sx
		for y in range(sy, 0, -1):
			if map[x, y] == 1:
				return (x, y)
	elif a > 0:
		for x in range(0, width - sx):
			y = sx + math.floor(x * math.tan(math.radians(a)))
			if map[x, y] == 1:
				return (x, y)
	elif a < 0:
		for x in range(0, sx):
			y = sx - math.floor(math.abs(x) * math.tan(math.radians(a)))
			if map[x, y] == 1:
				return (x, y)
	elif a == 0:
		x = sx
		for y in range(sy, height):
			if map[x, y] == 1:
				return (x, y)

def update():
	ultra_LS = raycast(bot.pos[0], bot.pos[0], bot.angle + 45)
	ultra_LC = raycast(bot.pos[0], bot.pos[0], bot.angle + 10)
	ultra_FC = raycast(bot.pos[0], bot.pos[0], bot.angle + 0)
	ultra_RC = raycast(bot.pos[0], bot.pos[0], bot.angle - 10)
	ultra_RS = raycast(bot.pos[0], bot.pos[0], bot.angle - 45)

	## start nural magic ##

	servo_FL = 1
	servo_FR = 1
	servo_BL = 1
	servo_BR = 1

	## end nutal magic ##

	pass
