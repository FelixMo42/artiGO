# prefs

ui = True
trail = True
events = True
duration = 1000
speed = 5

# imports

import tensorflow as tf
import numpy as np
import pygame as pg
import random
import math
import time

# create  map

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
	"angle" : 45,
	"color" : [0,0,255]
}

target = [750, 750]

# raycasting

def dist(xi,yi,xii,yii):
	sq1 = (xi-xii) ** 2
	sq2 = (yi-yii) ** 2
	return math.sqrt(sq1 + sq2)

def inrange(x,y):
	return x >= 0 and y >= 0 and x < width and y < height

def raycastcheak(sx,sy,x,y,d,color):
	if not inrange(x, y):
		return "break"

	if d != -1 and dist(sx, sy, x, y) >= d:
		return "break"

	if color != -1:
		screen.set_at((x,y), color)

	if map[x, y] == 1:
		if d != -1:
			return False
		return dist(sx, sy, x, y)

def raycast(sx,sy,a,d = -1,color = -1):
	sx = int(sx)
	sy = int(sy)

	a -= 90

	while a > 360:
		a -= 360
	while a < 0:
		a  += 360

	if a == 90 or a == -270:
		x = sx
		for y in range(0, sy):
			ret = raycastcheak(sx,sy,x,y,d,color)
			if ret == "break":
				break
			elif ret is not None:
				return ret
	elif a == -90 or a == 270:
		x = sx
		for y in range(sy, height):
			ret = raycastcheak(sx,sy,x,y,d,color)
			if ret == "break":
				break
			elif ret is not None:
				return ret
	elif a < 90 or a > 270:
		s = math.tan(math.radians(a))
		for x in range(sx, width):
			y = sy + math.floor((sx - x) * s)

			ret = raycastcheak(sx,sy,x,y,d,color)
			if ret == "break":
				break
			elif ret is not None:
				return ret
	elif a > 90 or a < 270:
		s = math.tan(math.radians(a))
		for x in range(0, sx):
			y = sy + math.floor((sx - x) * s)

			ret = raycastcheak(sx,sy,x,y,d,color)
			if ret == "break":
				break
			elif ret is not None:
				return ret
	if d == -1:
		return 100
	else:
		return True

# simulation

def events():
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.done = True
			
	if ui:
		pg.display.flip()

def run():
	bot["pos"] = [250,250]
	bot["angle"] = 45
	
	time = 1
	
	while not pg.done:
		if ui and not trail:
			draw((0,0,0))
		
		for i in range(speed):
			time += 1
			if not update() or time >= duration:
				# collison happened
				
				return (duration + width) - time - dist(bot["pos"][0], bot["pos"][1], target[0], target[1]) 
		
		if ui:
			draw(bot["color"])
		
		if events:
			events()
	

def update():
	# collison

	a = math.radians(-bot["angle"])
	sin = math.sin(a)
	cos = math.cos(a)

	w = bot["size"][0] / 2
	h = bot["size"][1] / 2
	
	if not raycast(
		bot["pos"][0] + -w * cos - h * sin,
		bot["pos"][1] + -w * sin + h * cos,
		bot["angle"] + 90, bot["size"][0]
	) or not raycast(
		bot["pos"][0] + -w * cos - h * sin,
		bot["pos"][1] + -w * sin + h * cos,
		bot["angle"] - 180, bot["size"][1]
	) or not raycast(
		bot["pos"][0] + w * cos - -h * sin,
		bot["pos"][1] + w * sin + -h * cos,
		bot["angle"] - 90, bot["size"][0]
	) or not raycast(
		bot["pos"][0] + w * cos - -h * sin,
		bot["pos"][1] + w * sin + -h * cos,
		bot["angle"], bot["size"][1]
	): return False
	
	#get input

	ultra_LS = raycast(bot["pos"][0], bot["pos"][0], bot["angle"] + 45)
	ultra_LC = raycast(bot["pos"][0], bot["pos"][0], bot["angle"] + 10)
	ultra_FC = raycast(bot["pos"][0], bot["pos"][0], bot["angle"] + 0)
	ultra_RC = raycast(bot["pos"][0], bot["pos"][0], bot["angle"] - 10)
	ultra_RS = raycast(bot["pos"][0], bot["pos"][0], bot["angle"] - 45)

	# nural magic

	servo_FL = random.randint(-1, 3)
	servo_FR = random.randint(-1, 3)
	servo_BL = random.randint(-1, 3)
	servo_BR = random.randint(-1, 3)

	# move bot

	LS = servo_FL + servo_BL
	RS = servo_FR + servo_BR

	if LS > RS:
		bot["angle"] += (LS - RS)
	elif RS > LS:
		bot["angle"] -= (RS - LS)

	speed = (LS + RS) / 4

	bot["pos"][0] += int(speed * math.sin(math.radians(bot["angle"])))
	bot["pos"][1] += int(speed * math.cos(math.radians(bot["angle"])))
	
	return True

# pygame setup

pg.init()
pg.done = False

# graphics

def draw(color):
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

if ui:
	screen = pg.display.set_mode((width, height))

	for x in range(width):
		for y in range(height):
			if map[x, y] == 1:
				screen.set_at((x,y), (255,255,255))

	pg.draw.circle(screen, (0,255,0), target, 5, 2)

# main loop

while not pg.done:
	print(run())
