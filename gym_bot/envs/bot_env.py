draws = 10
anglediv = 2
maxtime = 1000
trail = True

## imports ##

import gym
import numpy as np
import queue
import pygame as pg
import matplotlib.pyplot as plt
import math
import random
import time
import threading
import functools

## graph ##

plt.ion()

class Graph:
    def __init__(self, data, color = "blue"):
        self.fig = plt.figure()
        self.data = data
        self.color = color
        self.update()

    def update(self):
        plt.plot(self.data, color=self.color)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

data = []
gq = queue.Queue()
graph = Graph(data)

def plot(v):
    def func():
        data.append(v)
        graph.update()
    gq.put(func)

def pltUpdate(self):
    if not gq.empty():
        item = gq.get_nowait()
        item()
        gq.task_done()

gt = threading.Thread(target=pltUpdate)
gt.start()

## create  map ##

width = 1000
height = 1000

map = np.zeros((width,height), dtype=int)

def addRect(sx,sy,w,h):
    map[sx : sx + w,sy : sy + h] = 1

#addRect(500 - 30,500 - 30,60,60)

target = [750, 750]

## ray casting ##

def dist(xi,yi,xii,yii):
    sq1 = (xi-xii) ** 2
    sq2 = (yi-yii) ** 2
    return math.sqrt(sq1 + sq2)

def inrange(x,y):
    return x >= 0 and y >= 0 and x < width and y < height

def raycastcheak(sx,sy,x,y,d,color):
    if not inrange(x, y):
        return "break"

    if d != -1 and not inrange(x,y):
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

    if a == 90:
        x = sx
        for y in range(d and sy - d or 0, sy):
            ret = raycastcheak(sx,sy,x,y,d,color)
            if ret == "break":
                break
            elif ret is not None:
                return ret
    if a == -270:
        x = sx
        for y in range(sy, d and sy + d or height):
            ret = raycastcheak(sx,sy,x,y,d,color)
            if ret == "break":
                break
            elif ret is not None:
                return ret
    elif a == -90 or a == 270:
        x = sx
        for y in range(sy, d and sy + d or height):
            ret = raycastcheak(sx,sy,x,y,d,color)
            if ret == "break":
                break
            elif ret is not None:
                return ret
    elif a < 90 or a > 270:
        s = math.tan(math.radians(a))
        for x in range(sx, d and sx + d or width):
            y = sy + math.floor((sx - x) * s)

            ret = raycastcheak(sx,sy,x,y,d,color)
            if ret == "break":
                break
            elif ret is not None:
                return ret
    elif a > 90 or a < 270:
        s = math.tan(math.radians(a))
        for x in range(d and sx - d or 0, sx):
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

## graphics ##

pg.init()
screen = pg.display.set_mode((width, height))
q = queue.Queue()

background = pg.Surface([width, height])

for x in range(width):
    for y in range(height):
        if map[x, y] == 1:
            Background.set_at((x,y), (255,255,255))

pg.draw.circle(background, (0,255,0), target, 24, 2)

def pgUpdate():
    c = 0
    while True:
        item = q.get()
        screen.blit(background, (0,0), special_flags=pg.BLEND_MAX)
        item()
        q.task_done()

dt = threading.Thread(target=pgUpdate)
dt.start()

def draw(pos,w,h,a,color,line):
    a = math.radians(-a)
    sin = math.sin(a)
    cos = math.cos(a)

    w /= 2
    h /= 2

    pg.draw.lines(screen, color, True, [
        pos + np.array([-w * cos -  h * sin, -w * sin +  h * cos]),
        pos + np.array([-w * cos - -h * sin, -w * sin + -h * cos]),
        pos + np.array([ w * cos - -h * sin,  w * sin + -h * cos]),
        pos + np.array([ w * cos -  h * sin,  w * sin +  h * cos])
    ], line)

    pg.display.flip()

## gym ##

def rangeify(n):
    if n > 0 and n < 1:
        return 1
    elif n < 0 and n > -1:
        return -1
    else:
        return n

class BotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    pos = [250,250]
    size = [24, 60]
    angle = 45 + 180
    color = [0,0,255]

    total = 0
    time = 0
    p = 0
    sim = 0
    end = "None"

    updater = pltUpdate

    def __init__(self):
        pass

    def _step(self, action):
        ## at goal ##
        goal = False

        if dist(self.pos[0], self.pos[1], target[0], target[1]) < self.size[0]:
            goal = True
            self.end = "goal"

        ## hit thing ##

        collison = False

        a = math.radians(-self.angle)
        sin = math.sin(a)
        cos = math.cos(a)

        w = self.size[0] / 2
        h = self.size[1] / 2

        if not raycast(
            self.pos[0] + -w * cos - h * sin,
            self.pos[1] + -w * sin + h * cos,
            self.angle + 90, self.size[0]
        ) or not raycast(
            self.pos[0] + -w * cos - h * sin,
            self.pos[1] + -w * sin + h * cos,
            self.angle - 180, self.size[1]
        ) or not raycast(
            self.pos[0] + w * cos - -h * sin,
            self.pos[1] + w * sin + -h * cos,
            self.angle - 90, self.size[0]
        ) or not raycast(
            self.pos[0] + w * cos - -h * sin,
            self.pos[1] + w * sin + -h * cos,
            self.angle, self.size[1]
        ):
            collison = True
            self.end = "collison"

        ## out of bounds ##

        outofbounds = False

        '''
        if not inrange(self.pos[0], self.pos[1]):
            outofbounds = True
            self.end = "out of bounds"
        '''

        ## time ##

        self.time += 1

        timeout = False

        if self.time >= maxtime:
            timeout = True
            self.end = "time"

        ## update stuff ##

        done = goal or collison or outofbounds or timeout

        if not done:
            if not trail and self.time % draws == 0 and self.time != draws:
                q.put(self.clear)

            self._take_action(action)

            if self.time % draws == 0:
                q.put(self.drawFunc(self.color, 2))
                if not trail:
                    self.clear = self.drawFunc((0,0,0), 2)

        reward = self._get_reward(collison, goal)
        ob = self._get_info() #self.env.getState()

        self.total += reward

        return ob, reward, done, {}

    def _reset(self, sim):
        if self.time != 0:
            avg = int(self.total / self.time)
            print("sim: ", sim, "\t| score: ", avg, "\t| time: ", self.time, "\t| end: ", self.end)
            plot(min(avg,0))

        if not trail and self.time > draws:
            q.put(self.clear)

        self.sim = sim
        self.pos = [random.randint(0,1000),random.randint(0,1000)]
        self.angle = 45 + 180
        self.total = 0
        self.time = 0
        self.color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
        self.end = "none"

        return self._get_info()

    def _render(self, mode='human', close=False):
        pass

    def drawFunc(self, color, line):
        pos = [self.pos[0], self.pos[1]]
        w = self.size[0]
        h = self.size[1]
        a = self.angle

        def func():
            draw(pos, w, h, a, color, line)

        return func

    def _take_action(self, action):
        servo_FL, servo_FR, servo_BL, servo_BR = action

        servo_FL = rangeify(servo_FL)
        servo_FR = rangeify(servo_FR)
        servo_BL = rangeify(servo_BL)
        servo_BR = rangeify(servo_BR)

        LS = servo_FL + servo_BL
        RS = servo_FR + servo_BR

        if LS > RS:
            self.angle += (LS - RS) / (2 * anglediv)
        elif RS > LS:
            self.angle -= (RS - LS) / (2 * anglediv)

        speed = (LS + RS) / 4

        self.pos[0] += int(speed * math.sin(math.radians(self.angle)))
        self.pos[1] += int(speed * math.cos(math.radians(self.angle)))

    def _get_reward(self, collison, goal):
        reward = 0

        if collison:
            reward -= 100
        if goal:
            reward = 1000000000
        else:
            d = dist(self.pos[0], self.pos[1], target[0], target[1])
            reward -= d / 10
            #if d > dist(250,250 , target[0], target[1]):
            #    reward *= 2

        return reward

    def _get_info(self):
        ultra_LS = raycast(self.pos[0], self.pos[0], self.angle + 45) * 10
        ultra_LC = raycast(self.pos[0], self.pos[0], self.angle + 10) * 10
        ultra_FC = raycast(self.pos[0], self.pos[0], self.angle + 0) * 10
        ultra_RC = raycast(self.pos[0], self.pos[0], self.angle - 10) * 10
        ultra_RS = raycast(self.pos[0], self.pos[0], self.angle - 45) * 10

        return [
            ultra_LS, ultra_LC, ultra_FC, ultra_RC, ultra_RS,
            self.pos[0], self.pos[1], self.angle,
            target[0], target[1]
        ]
