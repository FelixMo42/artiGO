import gym
import numpy as np
import math
import random
import time

## create  map ##

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

target = [750, 750]

### ray casting ##

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

## gym ##

class BotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    pos = [250,250]
    size = [24, 60]
    angle = 45
    color = [0,0,255]

    total = 0

    def __init__(self):
        pass

    def _step(self, action):
        goal = False

        if dist(self.pos[0], self.pos[1], target[0], target[1]) < self.size[0]:
            goal = True

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

        done = target or collison
        if not done:
            self._take_action(action)
        reward = self._get_reward(collison, goal)
        ob = self._get_info() #self.env.getState()

        self.total += reward

        return ob, reward, done, {}

    def _reset(self):
        print(self.total)

        self.pos = [250,250]
        self.angle = 45
        self.total = 0

        return self._get_info()

    def _render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        servo_FL, servo_FR, servo_BL, servo_BR = action

        LS = servo_FL + servo_BL
        RS = servo_FR + servo_BR

        if LS > RS:
            self.angle += (LS - RS)
        elif RS > LS:
            self.angle -= (RS - LS)

        speed = (LS + RS) / 4

        self.pos[0] += int(speed * math.sin(math.radians(self.angle)))
        self.pos[1] += int(speed * math.cos(math.radians(self.angle)))

    def _get_reward(self, collison, goal):
        reward = 0

        if collison:
            reward = -1000
        elif goal:
            reward = 1000
        else:
            reward = (width - dist(self.pos[0], self.pos[1], target[0], target[1]))

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
