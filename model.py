import os
import math
import random
import numpy as np
import tensorflow as tf
from past.builtins import xrange


class MemN2N(object):
    def __init__(self, config, sess):
        self.nwords = config.nwords
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm

        self.show = config.show
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

        self.input = tf.placeholder(tf.float32, [None, self.edim], name="input")
        self.time = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
        self.target = tf.placeholder(tf.float32, [self.batch_size, self.nwords], name="target")
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="context")

        self.hid = []
        self.hid.append(self.input)
        self.share_list = []
        self.share_list.append([])

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

    def build_memory(self):
        self.global_step = tf.Variable(0, name="global_step")

        # 10000x150
        self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        # 150x150
        self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))

        # Temporal Encoding
        # 100x150
        self.T_A = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))
        self.T_B = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))

        # m_i = sum A_ij * x_ij + T_A_i
        # A: 10000x150, context: 128x100
        # Ain_c: 128x100x150
        Ain_c = tf.nn.embedding_lookup(self.A, self.context)
        # A: 10000x150, context: ?x100
        # Ain_t: ?x100x150
        Ain_t = tf.nn.embedding_lookup(self.T_A, self.time)
        # Ain: 128x100x150
        Ain = tf.add(Ain_c, Ain_t)

        # c_i = sum B_ij * u + T_B_i
        # B: 10000x150, context: 128x100
        # Bin_c: 128x100x50
        Bin_c = tf.nn.embedding_lookup(self.B, self.context)
        # T_B: 100x150, time: ?x100
        # Bin_c: 128x100x50
        Bin_t = tf.nn.embedding_lookup(self.T_B, self.time)
        # Bin: 128x100x50
        Bin = tf.add(Bin_c, Bin_t)

        for h in xrange(self.nhop):
            # hid3dim: ?x1x150
            self.hid3dim = tf.reshape(self.hid[-1], [-1, 1, self.edim])
            # Aout: 128x1x100
            Aout = tf.matmul(self.hid3dim, Ain, adjoint_b=True)
            # Aout2dim: 128x100, mem_size: 100
            Aout2dim = tf.reshape(Aout, [-1, self.mem_size])
            # 128x100
            P = tf.nn.softmax(Aout2dim)
            # probs3dim: 128x1x100
            probs3dim = tf.reshape(P, [-1, 1, self.mem_size])
            # Bout: 128x1x150
            Bout = tf.matmul(probs3dim, Bin)
            # Bout2dim: 128x150
            Bout2dim = tf.reshape(Bout, [-1, self.edim])
            # Cout: 128x150
            Cout = tf.matmul(self.hid[-1], self.C)
            # Dout: 128x150
            Dout = tf.add(Cout, Bout2dim)

            self.share_list[0].append(Cout)

            if self.lindim == self.edim:
                self.hid.append(Dout)
            elif self.lindim == 0:
                self.hid.append(tf.nn.relu(Dout))
            else:
                # F: 128x75, lindim: 75
                F = tf.slice(Dout, [0, 0], [self.batch_size, self.lindim])
                # G: 128x75
                G = tf.slice(Dout, [0, self.lindim], [self.batch_size, self.edim - self.lindim])
                # K: 128x75
                K = tf.nn.relu(G)
                self.hid.append(tf.concat(axis=1, values=[F, K]))

    def build_model(self):
        self.build_memory()

        # W: 150x10000
        self.W = tf.Variable(tf.random_normal([self.edim, self.nwords], stddev=self.init_std))
        # Z: 128x10000
        z = tf.matmul(self.hid[-1], self.W)
        # loss shape: (128,)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=self.target)

        self.lr = tf.Variable(self.current_lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W]
        grads_and_vars = self.opt.compute_gradients(self.loss, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def train(self, data):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords])  # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            time[:, t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=N)

        for idx in xrange(N):
            if self.show: bar.next()
            target.fill(0)
            for b in xrange(self.batch_size):
                m = random.randrange(self.mem_size, len(data))
                target[b][data[m]] = 1
                context[b] = data[m - self.mem_size:m]

            _, loss, self.step = self.sess.run([self.optim,
                                                self.loss,
                                                self.global_step],
                                               feed_dict={
                                                   self.input: x,
                                                   self.time: time,
                                                   self.target: target,
                                                   self.context: context})
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost / N / self.batch_size

    def test(self, data, label='Test'):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords])  # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            time[:, t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar(label, max=N)

        m = self.mem_size
        for idx in xrange(N):
            if self.show: bar.next()
            target.fill(0)
            for b in xrange(self.batch_size):
                target[b][data[m]] = 1
                context[b] = data[m - self.mem_size:m]
                m += 1

                if m >= len(data):
                    m = self.mem_size

            loss = self.sess.run([self.loss], feed_dict={self.input: x,
                                                         self.time: time,
                                                         self.target: target,
                                                         self.context: context})
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost / N / self.batch_size

    def run(self, train_data, test_data):
        if not self.is_test:
            for idx in xrange(self.nepoch):
                train_loss = np.sum(self.train(train_data))
                test_loss = np.sum(self.test(test_data, label='Validation'))

                # Logging
                self.log_loss.append([train_loss, test_loss])
                self.log_perp.append([math.exp(train_loss), math.exp(test_loss)])

                state = {
                    'perplexity': math.exp(train_loss),
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'valid_perplexity': math.exp(test_loss)
                }
                print(state)

                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx - 1][1] * 0.9999:
                    self.current_lr = self.current_lr / 1.5
                    self.lr.assign(self.current_lr).eval()
                if self.current_lr < 1e-5: break

                if idx % 10 == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                    global_step=self.step.astype(int))
        else:
            self.load()

            valid_loss = np.sum(self.test(train_data, label='Validation'))
            test_loss = np.sum(self.test(test_data, label='Test'))

            state = {
                'valid_perplexity': math.exp(valid_loss),
                'test_perplexity': math.exp(test_loss)
            }
            print(state)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")
