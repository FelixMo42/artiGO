OUTPUT_GRAPH = False
MAX_EPISODE = 1000
MAX_EP_STEPS = 200
DISPLAY_REWARD_THRESHOLD = -100  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time
GAMMA = 0.9
LR_A = 0.001 # learning rate for actor
LR_C = 0.01 # learning rate for critic

## prefs ##

ui = True
trail = True
events = True

duration = 2000
speed = 10

width = 1000
height = 1000

## imports ##

import tensorflow as tf
import numpy as np
import pygame as pg

import random
import math
import time
import threading
import multiprocessing

np.random.seed(2)
tf.set_random_seed(2)

## create  map ##

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

## creat bot ##

# actor in AC

class Actor(object):
	def __init__(self, sess, n_features, n_actions, action_bound, lr=0.0001):
		self.sess = sess

		self.s = tf.placeholder(tf.float32, [1, n_features], "state")
		self.a = tf.placeholder(tf.float32, None, name="act")
		self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error

		l1 = tf.layers.dense(
			inputs=self.s,
			units=30,  # number of hidden units
			activation=tf.nn.relu,
			kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
			bias_initializer=tf.constant_initializer(0.1),  # biases
			name='l1'
		)

		mu = tf.layers.dense(
			inputs=l1,
			units=1,  # number of hidden units
			activation=tf.nn.tanh,
			kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
			bias_initializer=tf.constant_initializer(0.1),  # biases
			name='mu'
		)

		sigma = tf.layers.dense(
			inputs=l1,
			units=1,  # output units
			activation=tf.nn.softplus,  # get action probabilities
			kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
			bias_initializer=tf.constant_initializer(1.),  # biases
			name='sigma'
		)
		global_step = tf.Variable(0, trainable=False)
		# self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
		self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma+0.1)
		self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

		self.action = tf.clip_by_value(self.normal_dist.sample(n_actions), action_bound[0], action_bound[1])

		with tf.name_scope('exp_v'):
			log_prob = self.normal_dist.log_prob(self.a)  # loss without advantage
			self.exp_v = log_prob * self.td_error  # advantage (TD_error) guided loss
			# Add cross entropy cost to encourage exploration
			self.exp_v += 0.01*self.normal_dist.entropy()

		with tf.name_scope('train'):
			self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)	# min(v) = max(-v)

	def learn(self, s, a, td):
		feed_dict = {self.s: s, self.a: a, self.td_error: td}
		_, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
		return exp_v

	def choose_action(self, s):
		return self.sess.run(self.action, {self.s: s})  # get probabilities for all actions


class Critic(object):
	def __init__(self, sess, n_features, lr=0.01):
		self.sess = sess
		with tf.name_scope('inputs'):
			self.s = tf.placeholder(tf.float32, [1, n_features], "state")
			self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next")
			self.r = tf.placeholder(tf.float32, name='r')

		with tf.variable_scope('Critic'):
			l1 = tf.layers.dense(
				inputs=self.s,
				units=30,  # number of hidden units
				activation=tf.nn.relu,
				kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
				bias_initializer=tf.constant_initializer(0.1),  # biases
				name='l1'
			)

			self.v = tf.layers.dense(
				inputs=l1,
				units=1,  # output units
				activation=None,
				kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
				bias_initializer=tf.constant_initializer(0.1),  # biases
				name='V'
			)

		with tf.variable_scope('squared_TD_error'):
			self.td_error = tf.reduce_mean(self.r + GAMMA * self.v_ - self.v)
			self.loss = tf.square(self.td_error)	# TD_error = (r+gamma*V_next) - V_eval
		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

	def learn(self, s, r, s_):
		v_ = self.sess.run(self.v, {self.s: s_})
		td_error, _ = self.sess.run([self.td_error, self.train_op],
										  {self.s: s, self.v_: v_, self.r: r})
		return td_error

tf.reset_default_graph()

## simulation ##

def events():
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.done = True

	if ui:
		pg.display.flip()

def info():
	ultra_LS = raycast(bot.pos[0], bot.pos[0], bot.angle + 45) * 10
	ultra_LC = raycast(bot.pos[0], bot.pos[0], bot.angle + 10) * 10
	ultra_FC = raycast(bot.pos[0], bot.pos[0], bot.angle + 0) * 10
	ultra_RC = raycast(bot.pos[0], bot.pos[0], bot.angle - 10) * 10
	ultra_RS = raycast(bot.pos[0], bot.pos[0], bot.angle - 45) * 10

	return [[
		ultra_LS, ultra_LC, ultra_FC, ultra_RC, ultra_RS,
		bot.pos[0], bot.pos[1],
		target[0], target[1]
	]]

def run():
	bot.pos = [250,250]
	bot.angle = 45

	time = 1

	while not pg.done:
		if ui and not trail:
			draw((0,0,0))

		init = info()

		for i in range(speed):
			time += 1
			if not update(init, time) or time >= duration:
				return (duration + width) - (time + dist(bot.pos[0], bot.pos[1], target[0], target[1]))

		if ui:
			draw(bot.color)

		if events:
			events()

def update(init, t):
	# collison

	a = math.radians(-bot.angle)
	sin = math.sin(a)
	cos = math.cos(a)

	w = bot.size[0] / 2
	h = bot.size[1] / 2

	if not raycast(
		bot.pos[0] + -w * cos - h * sin,
		bot.pos[1] + -w * sin + h * cos,
		bot.angle + 90, bot.size[0]
	) or not raycast(
		bot.pos[0] + -w * cos - h * sin,
		bot.pos[1] + -w * sin + h * cos,
		bot.angle - 180, bot.size[1]
	) or not raycast(
		bot.pos[0] + w * cos - -h * sin,
		bot.pos[1] + w * sin + -h * cos,
		bot.angle - 90, bot.size[0]
	) or not raycast(
		bot.pos[0] + w * cos - -h * sin,
		bot.pos[1] + w * sin + -h * cos,
		bot.angle, bot.size[1]
	): return False

	# nural magic

	input = info()
	output = bot.choose_action(input)

	servo_FL, servo_FR, servo_BL, servo_BR = output

	reward = (t - duration) + (width - dist(bot.pos[0], bot.pos[1], target[0], target[1]))

	td_error = critic.learn(init, reward, input)
	bot.learn(init, output, td_error)

	# move bot

	LS = servo_FL + servo_BL
	RS = servo_FR + servo_BR

	if LS > RS:
		bot.angle += (LS - RS)
	elif RS > LS:
		bot.angle -= (RS - LS)

	speed = (LS + RS) / 4

	bot.pos[0] += int(speed * math.sin(math.radians(bot.angle)))
	bot.pos[1] += int(speed * math.cos(math.radians(bot.angle)))

	return True

## pygame setup ##

if ui or events:
	pg.init()
	pg.done = False
	screen = pg.display.set_mode((width, height))

## graphics ##

def draw(color):
	a = math.radians(-bot.angle)
	sin = math.sin(a)
	cos = math.cos(a)

	w = bot.size[0] / 2
	h = bot.size[1] / 2

	pg.draw.lines(screen, color, True, [
		bot.pos + np.array([-w * cos -  h * sin, -w * sin +  h * cos]),
		bot.pos + np.array([-w * cos - -h * sin, -w * sin + -h * cos]),
		bot.pos + np.array([ w * cos - -h * sin,  w * sin + -h * cos]),
		bot.pos + np.array([ w * cos -  h * sin,  w * sin +  h * cos])
	])

if ui:
	for x in range(width):
		for y in range(height):
			if map[x, y] == 1:
				screen.set_at((x,y), (255,255,255))

	pg.draw.circle(screen, (0,255,0), target, 5, 2)

## main loop ##

with tf.Session() as sess:

	bot = Actor(sess, n_features=9, n_actions=4, lr=0.01, action_bound=[-255, 255])
	critic = Critic(sess, n_features=9, lr=0.01)
	bot.pos = [250,250]
	bot.size = [24, 60]
	bot.angle = 45
	bot.color = [0,0,255]

	init = tf.global_variables_initializer()

	sess.run(init)

	while not pg.done:
		print(run())
