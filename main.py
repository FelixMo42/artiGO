# imports

import tensorflow as tf
import numpy as np
import pygame as pg
import random
import math

# creat  map

width = 1000
height = 1000

map = np.zeros((width,height), dtype=int)
map[:,0] = 1
map[:,height - 1] = 1
map[0,:] = 1
map[width - 1,:] = 1

def addRect(sx,sy,w,h):
	map[sx : sx + w,sy : sy + h] = 1

addRect(500 - 30,500 - 30,60,60)

# creat simulation

bot = {
	"pos" : [250, 250],
	"size" : [24, 60],
	"angle" : 45
}

target = [750, 750]

def dist(xi,yi,xii,yii):
	sq1 = (xi-xii)*(xi-xii)
	sq2 = (yi-yii)*(yi-yii)
	return math.sqrt(sq1 + sq2)

def raycast(sx,sy,a):
	if a == 180:
		x = sx
		for y in range(sy, 0, -1):
			if map[x, y] == 1:
				return dist(sx, sy, x, y)
	elif a > 0:
		for x in range(0, width - sx):
			y = sx + math.floor(x * math.tan(math.radians(a)))
			if map[x, y] == 1:
				return dist(sx, sy, x, y)
	elif a < 0:
		for x in range(0, sx):
			y = sx - math.floor(math.fabs(x) * math.tan(math.radians(a)))
			if map[x, y] == 1:
				return dist(sx, sy, x, y)
	elif a == 0:
		x = sx
		for y in range(sy, height):
			if map[x, y] == 1:
				return dist(sx, sy, x, y)

def update():
	ultra_LS = raycast(bot["pos"][0], bot["pos"][0], bot["angle"] + 45)
	ultra_LC = raycast(bot["pos"][0], bot["pos"][0], bot["angle"] + 10)
	ultra_FC = raycast(bot["pos"][0], bot["pos"][0], bot["angle"] + 0)
	ultra_RC = raycast(bot["pos"][0], bot["pos"][0], bot["angle"] - 10)
	ultra_RS = raycast(bot["pos"][0], bot["pos"][0], bot["angle"] - 45)

	## start nural magic ##

	servo_FL = random.randint(-1, 3)
	servo_FR = random.randint(-1, 3)
	servo_BL = random.randint(-1, 3)
	servo_BR = random.randint(-1, 3)

	## end nutal magic ##

	LS = servo_FL + servo_BL
	RS = servo_FR + servo_BR

	if LS > RS:
		bot["angle"] += (LS - RS)
	elif RS > LS:
		bot["angle"] -= (RS - LS)

	speed = (LS + RS) / 4

	bot["pos"][0] += int(speed * math.sin(math.radians(bot["angle"])))
	bot["pos"][1] += int(speed * math.cos(math.radians(bot["angle"])))

## pygame stuff ##

pg.init()
screen = pg.display.set_mode((width, height))
done = False

# draw map

for x in range(width):
	for y in range(height):
		if map[x, y] == 1:
			screen.set_at((x,y), (255,255,255))

pg.draw.circle(screen, (0,255,0), target, 5, 2)

# draw bot

def draw(color):
	pg.transform.rotate(screen, bot["angle"])

	a = math.radians(-bot["angle"])
	sin = math.sin(a)
	cos = math.cos(a)

	w = bot["size"][0] / 2
	h = bot["size"][1] / 2

	pg.draw.lines(screen, color, True, [
		bot["pos"] + np.array([-w * cos -  h * sin, -w * sin +  h * cos]),
		bot["pos"] + np.array([-w * cos - -h * sin, -w * sin + -h * cos]),
		bot["pos"] + np.array([ w * cos - -h * sin,  w * sin + -h * cos]),
		bot["pos"] + np.array([ w * cos -  h * sin,  w * sin +  h * cos])
	])

# main loop

while not done:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			done = True

	draw((0,0,0))
	update()
	draw((0,0,255))

	pg.display.flip()
