draws = 15
anglediv = 10
maxtime = 500
trail = False
graphics = True
graphing = True
avg = 100

## imports ##

import gym
import numpy as np
from queue import Queue
import pygame as pg
import matplotlib.pyplot as plt
import math
import random
import time
import threading
import functools

## graph ##

if graphing:
    plt.ion()

    class Graph:
        def __init__(self, datas, color = "blue"):
            self.fig = plt.figure()
            self.datas = datas
            self.color = color
            self.update()

        def update(self):
            for data in self.datas:
                if len(data["data"]) > 0:
                    plt.plot(np.arange(len(data["data"])) * data["tick"], data["data"], color=data["color"], linewidth=data["size"])

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    gq = Queue()

    pltAvgs = [0]
    pltAvgs10 = [0]
    pltAvgs100 = [0]
    pltAvgs1000 = [0]

    pltWins = [0]
    pltWins10 = [0]
    pltWins100 = [0]
    pltWins1000 = [0]

    graph = Graph([
        #{"data": pltAvgs, "color": "blue", "size": 1, "tick": 1},
        #{"data": pltAvgs10, "color": "blue", "size": 2, "tick": 10},
        #{"data": pltAvgs100, "color": "blue", "size": 3, "tick": 100},
        #{"data": pltAvgs1000, "color": "blue", "size": 4, "tick": 1000},
        #{"data": pltWins, "color": "yellow", "size": 1, "tick": 1},
        #{"data": pltWins10, "color": "yellow", "size": 2, "tick": 10},
        {"data": pltWins100, "color": "yellow", "size": 3, "tick": 100},
        {"data": pltWins1000, "color": "yellow", "size": 4, "tick": 1000},
    ])

    graph.count = 1

    graph.pltAvgs = pltAvgs
    graph.pltAvgs10 = pltAvgs10
    graph.pltAvgs100 = pltAvgs100
    graph.pltAvgs1000 = pltAvgs1000

    graph.pltWins = pltWins
    graph.pltWins10 = pltWins10
    graph.pltWins100 = pltWins100
    graph.pltWins1000 = pltWins1000

    def plot(v, s):
        def func():
            pltAvgs.append(v)
            #if graph.count % 10 == 0:
            #    graph.pltAvgs10.append(sum(graph.pltAvgs[-10:])/min(10, graph.count + 1))
            #if graph.count % 100 == 0:
            #    graph.pltAvgs100.append(sum(graph.pltAvgs[-100:])/min(100, graph.count + 1))
            #if graph.count % 1000 == 0:
            #    graph.pltAvgs1000.append(sum(graph.pltAvgs[-1000:])/min(1000, graph.count + 1))

            if s:
                pltWins.append(100.0)
            else:
                pltWins.append(0)
            #if graph.count % 10 == 0:
            #    graph.pltWins10.append(sum(graph.pltWins[-10:])/min(10, graph.count + 1))
            if graph.count % 100 == 0:
                graph.pltWins100.append(sum(graph.pltWins[-100:])/100)
            if graph.count % 1000 == 0:
                graph.pltWins1000.append(sum(graph.pltWins[-1000:])/1000)

            graph.count += 1
        gq.put(func)

    def pltUpdate():
        while not gq.empty():
            item = gq.get_nowait()
            item()
            gq.task_done()
        graph.update()

    gt = threading.Thread(target=pltUpdate)
    gt.start()

## create  map ##

width = 1000
height = 1000

map = np.zeros((width,height), dtype=int)

def addRect(sx,sy,w,h):
    map[sx : sx + w,sy : sy + h] = 1

#addRect(500 - 30,500 - 30,60,60)

target = [500, 500]

## ray casting ##

sqrts = {}

def dist(xi,yi,xii,yii):
    pow = (xi-xii) ** 2 + (yi-yii) ** 2
    if pow not in sqrts:
        sqrts[pow] = math.sqrt(pow)
    return sqrts[pow]

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

if graphics:
    pg.init()
    screen = pg.display.set_mode((width, height))
    q = Queue()

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
        sin = math.sin(-a)
        cos = math.cos(-a)

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
    return n / 255 * 10

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
    end = "none"

    dist = dist

    def __init__(self):
        if graphing:
            self.updater = pltUpdate

    def _step(self, action):
        ## at goal ##

        if dist(self.pos[0], self.pos[1], target[0], target[1]) < self.size[0]:
            self.end = "goal"
            self.goal = True

        ## hit thing ##

        '''a = math.radians(-self.angle)
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
            self.end = "collison"'''

        ## time ##

        self.time += 1

        if self.time >= maxtime:
            self.end = "time"

        ## update stuff ##

        done = self.end != "none" and self.end != "goal"

        if not done:
            if graphics and not trail and self.time % draws == 0 and self.time != draws:
                q.put(self.clear)

            self._take_action(action)

            if graphics and self.time % draws == 0:
                q.put(self.drawFunc(self.color, 2))
                if not trail:
                    self.clear = self.drawFunc((0,0,0), 2)

        reward = self._get_reward()
        ob = self._get_info()

        self.total += reward

        return ob, reward, done, {}

    def _reset(self, sim):
        if self.time != 0:
            avg = int(self.total / self.time)
            print("sim: ", sim, "\t| score: ", avg, "\t| time: ", self.time, "\t| goal: ", self.goal)

            if graphing:
                plot(avg, self.goal)

        if graphics and not trail and self.time > draws:
            q.put(self.clear)

        self.sim = sim
        self.goal = False
        a = math.radians(random.randint(1,360))
        self.pos = [500 * math.cos(a) + 500, 500 * math.sin(a) + 500]
        self.angle = 45 + 180
        self.total = 0
        self.time = 0
        self.color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
        self.end = "none"
        self.p = dist(self.pos[0], self.pos[1], target[0], target[1])

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

    def _get_reward(self):
        self.reward = -100

        d = dist(self.pos[0], self.pos[1], target[0], target[1])

        if d < self.size[0]:
            self.reward += 200
        elif self.goal:
            self.reward = 100 / d
        else:
            self.reward += max( (self.p - d) * 10, 0 )
            self.p = min(self.p, d)

        return self.reward

    def _get_info(self):
        #ultra_LS = raycast(self.pos[0], self.pos[0], self.angle + 45) * 10
        #ultra_LC = raycast(self.pos[0], self.pos[0], self.angle + 10) * 10
        #ultra_FC = raycast(self.pos[0], self.pos[0], self.angle + 0) * 10
        #ultra_RC = raycast(self.pos[0], self.pos[0], self.angle - 10) * 10
        #ultra_RS = raycast(self.pos[0], self.pos[0], self.angle - 45) * 10

        d = dist(self.pos[0], self.pos[1], target[0], target[1])
        angle_offset = self.angle - math.asin((self.pos[1] - target[1]) / d)

        return [
            #ultra_LS, ultra_LC, ultra_FC, ultra_RC, ultra_RS,
            self.pos[0], self.pos[1], d,
            target[0], target[1], angle_offset,
#            math.cos(self.angle), math.sin(self.angle),
#            (target[0] - self.pos[0]) * math.cos(math.radians(self.angle)) + (target[1] - self.pos[1]) * math.sin(math.radians(self.angle))
        ]
