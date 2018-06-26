#imports

import tensorflow as tf
import numpy as np
import pygame as pg
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

	servo_FL = 1
	servo_FR = 1
	servo_BL = 1
	servo_BR = 1

	## end nutal magic ##

	LS = servo_FL + servo_BL
	RS = servo_FR + servo_BR

	if LS > RS:
		bot["angle"] += 0
	elif RS > LS:
		bot["angle"] -= 0

	speed = (LS + RS) / 4

	bot["pos"][0] += int(speed * math.cos(math.radians(bot["angle"])))
	bot["pos"][1] += int(speed * math.sin(math.radians(bot["angle"])))

## pygame stuff ##

pg.init()
screen = pg.display.set_mode((width, height))
done = False

# draw map

for x in range(width):
	for y in range(height):
		if map[x, y] == 1:
			screen.set_at((x,y), [255,255,255])

# draw bot

def draw():
	pg.draw.rect(screen, [0,0,255], (bot["pos"] - np.array(bot["size"]) / 2, bot["size"] ), 2 )

def clear():
	pg.draw.rect(screen, [0,0,0], (bot["pos"] - np.array(bot["size"]) / 2, bot["size"] ), 2 )

# main loop

while not done:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			done = True

	clear()
	update()
	draw()

	pg.display.flip()
