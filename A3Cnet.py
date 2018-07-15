import multiprocessing
import threading
import numpy as np
import tensorflow as tf
import ctypes
import gym
import tensorlayer as tl
from tensorlayer.layers import DenseLayer, InputLayer
import time
import gym_bot

tf.set_random_seed(42)

GAME = 'Bot-v0'
OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 30
GAMMA = 0.995
ENTROPY_BETA = 0.005
LR_A = 0.0002 # learning rate for actor
LR_C = 0.001 # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0  # will increase during training, stop training when it >= MAX_GLOBAL_EP
l2_scale = 0.001
hidden = [32, 32]

env = gym.make(GAME)

N_S = len( env._get_info() )
N_A = 4
A_BOUND = [-5, 5]

class ACNet(object):
    def __init__(self, scope, globalAC=None):
        self.scope = scope
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net()
                self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
                self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)

                normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

                with tf.name_scope('choose_a'):
                    self.A = tf.nn.tanh(tf.squeeze(normal_dist.sample(1), axis=0)) * 255

        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                    self.c_loss_summary = tf.summary.scalar('{}-C-Loss'.format(scope), self.c_loss)

                with tf.name_scope('wrap_a_out'):
                    self.test = self.sigma[0]
                    self.mu, self.sigma = self.mu * A_BOUND[1], self.sigma + 1e-5

                normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.nn.tanh(tf.squeeze(normal_dist.sample(1), axis=0)) * 255

                with tf.name_scope('local_grad'):
                    self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
                    self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        w_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('actor'):
            regs = None
            activ = tf.nn.tanh

            nn = tf.layers.dense(self.s, units=hidden[0], activation=activ, kernel_initializer=w_init, name='la')

            # WTF! why does this break, is it becaus of relu
            nn = tf.layers.dense(nn, units=hidden[1], activation=activ, kernel_initializer=w_init, name='la2')
            # AHHHHHHHHHH

            self.mu = tf.layers.dense(nn, units=N_A, activation=tf.nn.tanh, kernel_initializer=w_init, kernel_regularizer=regs, name='mu')
            self.sigma = tf.layers.dense(nn, units=N_A, activation=tf.nn.softplus, kernel_initializer=w_init, kernel_regularizer=regs, name='sigma')

        with tf.variable_scope('critic'):
            nn = tf.layers.dense(self.s, units=hidden[0], activation=activ, kernel_initializer=w_init, kernel_regularizer=regs, name='lc')
            nn = tf.layers.dense(nn, units=hidden[1], activation=activ, kernel_initializer=w_init, kernel_regularizer=regs, name='lc2')
            self.v = tf.layers.dense(nn, units=1, kernel_initializer=w_init, name='v')

    def update_global(self, feed_dict):  # run by a local
        sess.run(self.c_loss_summary, feed_dict)
        _, _, t = sess.run([self.update_a_op, self.update_c_op, self.test], feed_dict)  # local grads applies to global net
        return t

    def pull_global(self):  # run by a local
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = [s]
        r = sess.run([self.A], {self.s: s})
        return r[0][0]

    def save_ckpt(self):
        tl.files.exists_or_mkdir(self.scope)
        tl.files.save_ckpt(
            sess=sess, mode_name='model.ckpt', var_list=self.a_params + self.c_params, save_dir=self.scope,
            printable=True
        )

    def load_ckpt(self):
        tl.files.load_ckpt(sess=sess, var_list=self.a_params + self.c_params, save_dir=self.scope, printable=True)
        # tl.files.load_ckpt(sess=sess, mode_name='model.ckpt', var_list=self.a_params+self.c_params, save_dir=self.scope, is_latest=False, printable=True)


class Worker(object):
    def __init__(self, i, globalAC):
        with tf.device("/cpu:" + str(i)):
            self.env = gym.make(GAME)
            self.name = 'Worker_%i' % i
            self.AC = ACNet(self.name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP

        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not threadsDone:
            s = self.env._reset(GLOBAL_EP)
            ep_r = 0
            while True:
                a = self.AC.choose_action(s)
                s_, r, done, _info = self.env.step(a)

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = sess.run(self.AC.v, {self.AC.s: [s_]})[0, 0]

                    buffer_v_target = []

                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = (
                        np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    )
                    feed_dict = {self.AC.s: buffer_s, self.AC.a_his: buffer_a, self.AC.v_target: buffer_v_target}
                    # update gradients on global network
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    # update local network from global network
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    with tf.device("/cpu:0"):
        OPT_A = tf.train.AdagradOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.AdagradOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
        workers = []
    for i in range(N_WORKERS):
        workers.append(Worker(i, GLOBAL_AC))

    sess = tf.Session(config=tf.ConfigProto(device_count={"CPU": N_WORKERS}))
    tl.layers.initialize_global_variables(sess)


    summary_writer = tf.summary.FileWriter('./summary',sess.graph)
    threadsDone = False
    worker_threads = []

    for worker in workers:
        t = threading.Thread(target=worker.work)
        t.start()
        worker_threads.append(t)

    try:
        while True:
            if hasattr(env,"updater"):
                env.updater()
            time.sleep(10)
    except KeyboardInterrupt:
        pass

    threadsDone = True
    for process in worker_threads:
        process.join()

    GLOBAL_AC.save_ckpt()

    # ============================= EVALUATION =============================
    # env = gym.make(GAME)
    # GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
    # tl.layers.initialize_global_variables(sess)
    # GLOBAL_AC.load_ckpt()
    # while True:
    #     s = env.reset()
    #     rall = 0
    #     while True:
    #         env.render()
    #         a = GLOBAL_AC.choose_action(s)
    #         s, r, d, _ = env.step(a)
    #         rall += r
    #         if d:
    #             prin("reward", rall)
    #             break
