# https://github.com/han-cai/rlb-dp/blob/master/python/bid_ls.py

# https://github.com/han-cai/rlb-dp/blob/master/python/lin_bid.py

from bidding_agent import bidding_agent
import pickle
import os
import time
from utils import getTime
from utils import write_log
from utils import load_data
from utils import evaluate_rmse
from utils import str_list2float_list
from utils import activate_calc
import sys
from bidding_agent_rtb_rl_fa_NN_Approximator import NN_Approximator
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error




class bidding_agent_rtb_rl_fa(bidding_agent):



    def init(self,  environment, camp_info, gamma, opt_obj):
        self.opt_obj = opt_obj
        self.gamma = gamma
        self.camp_info = camp_info
        self.v1 = self.opt_obj.v1
        self.v0 = self.opt_obj.v0

        self.D_info = []
        self.D_point = []
        self.N_bound = 0
        self.B_bound = 0

        self.nn_approx = None
        self.sess = None
        self.dim = 0

        self.net_type = None
        self.net_argv = None
        self.params = []
        super(bidding_agent_rtb_rl_fa, self).init(environment, camp_info)




    def approximate(self, model, src, camp, N, D_function_path,  large_storage_folder , NN_model_path):

        seeds = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC,
                 0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]

        if model == "dnb":
            dim = 2
            net_type = "nn"



            obj_type = "clk"
            clk_vp = 1
            tag = src + "_" + camp + "_" + model + "_" + net_type + "_N={}_{}".format(N, obj_type) + "_" + getTime()


            opt_obj = self.opt_obj
            camp_info = self.camp_info
            avg_theta = camp_info["clk_train"] / camp_info["imp_train"]
            if obj_type == "profit":
                avg_theta *= opt_obj.clk_v

            b_bound = 800
            n_bound = 50
            max_train_round = 100
            final_model_path = NN_model_path

            n_sample_size = 50
            b_sample_size = 200
            eval_n_sample_size = 500
            eval_b_sample_size = 1000
            batch_size = n_sample_size * b_sample_size

            net_argv = [4, [dim, 30, 15, 1], "tanh"]
            init_rag = avg_theta
            nn_approx = NN_Approximator(net_type, net_argv,
                                        # data_path + camp + "/bid-model/fa_dnb_{}.pickle".format(obj_type)
                                        None
                                        ,
                                        [('uniform', -0.001, 0.001, seeds[4]),
                                         ('zero', None),
                                         ('uniform', -0.001, 0.001, seeds[5]),
                                         ('zero', None),
                                         ('uniform', -init_rag, init_rag, seeds[6]),
                                         ('zero', None)
                                         ],
                                        [dim], batch_size,
                                        ['adam', 3e-5, 1e-8, 'sum']
                                        # ["ftrl", 1e-2, "mean"]
                                        # ['sgd', 2e-2, 'mean']
                                        )

            # need to split the D_function_path into one line files.
            train_dir = large_storage_folder + "/../fa-train/rlb_dnb_gamma=1_N={}_{}_1/".format(N, obj_type)
            #print("train_dir = {}".format(train_dir))
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
                with open(D_function_path, "r") as fin:
                    count = 0
                    for line in fin:
                        save_path = train_dir + str(count) + ".txt"
                        print(save_path)
                        with open(save_path, "w") as fout:
                            fout.write(line + "\n")
                        count += 1

            n_list = [i for i in range(n_bound + 1, N)]

            #print("n_list = {}".format(n_list))

            # train, eval
            mode = "train"
            save_model = True
            model_path = large_storage_folder + "../fa-train/" + tag + "/"
            log_path = large_storage_folder + "../fa-log/" + tag + ".txt"
            log_folder = large_storage_folder + "../fa-log/"
            if save_model and mode == "train":
                os.mkdir(model_path)

            if not os.path.exists(log_folder):
                os.mkdir(log_folder)

            #print(tag)
            #print(nn_approx.log)

            if mode == "train":
                if save_model:
                    write_log(log_path, nn_approx.log)

                with tf.Session(graph=nn_approx.graph) as sess:
                    tf.initialize_all_variables().run()
                    print("model initialized")

                    _iter = 0
                    perf = 1e5
                    start_time = time.time()
                    while True:
                        _iter += 1
                        print("iteration {0} start".format(_iter))
                        np.random.shuffle(n_list)

                        buf_loss = []
                        buf_predictions = []
                        buf_labels = []

                        _round = int(len(n_list) / n_sample_size)
                        #print("_round = {}".format(_round))
                        for _i in range(_round):
                            batch_n = n_list[_i * n_sample_size: (_i + 1) * n_sample_size]
                            batch_x_vecs, batch_value_labels = load_data(train_dir, batch_n, b_sample_size, b_bound,
                                                                         dim)

                            feed_dict = {
                                nn_approx.batch_x_vecs: batch_x_vecs,
                                nn_approx.batch_value_labels: batch_value_labels
                            }

                            _, loss, batch_predictions = sess.run([nn_approx.opt_value, nn_approx.loss_value,
                                                                   nn_approx.batch_value_predictions],
                                                                  feed_dict=feed_dict)
                            buf_loss.append(np.sqrt(loss) / avg_theta)
                            buf_labels.extend(batch_value_labels.flatten())
                            buf_predictions.extend(batch_predictions.flatten())
                        buf_loss = np.array(buf_loss)
                        #print("buff_loss = {}".format(buf_loss))
                        #print("buff_predictions = {}".format(buf_predictions))
                        buf_rmse = np.sqrt(mean_squared_error(buf_labels, buf_predictions))
                        buf_log = "buf loss, max={:.6f}\tmin={:.6f}\tmean={:.6f}\tbuf rmse={}\ttime={}".format(
                            buf_loss.max(), buf_loss.min(), buf_loss.mean(), buf_rmse / avg_theta, getTime())
                        print(buf_log)

                        np.random.shuffle(n_list)
                        eval_rmse = evaluate_rmse(train_dir, n_list[:eval_n_sample_size], eval_b_sample_size,
                                                  batch_size,
                                                  b_bound, dim, nn_approx)
                        eval_log = "iteration={}\ttime={}\teval rmse={}\tbuf rmse={}" \
                            .format(_iter, time.time() - start_time, eval_rmse / avg_theta, buf_rmse / avg_theta)
                        print(eval_log)
                        if save_model:
                            write_log(log_path, eval_log)
                            nn_approx.dump(model_path + "{}_{}.pickle".format(tag, _iter), net_type, net_argv)
                            n_perf = (buf_rmse + eval_rmse) / avg_theta
                            if n_perf < perf:
                                perf = n_perf
                                nn_approx.dump(final_model_path, net_type, net_argv)
                        start_time = time.time()
                        if _iter >= max_train_round:
                            nn_approx.dump(final_model_path, net_type, net_argv)
                            break

            elif mode == "eval":
                with tf.Session(graph=nn_approx.graph) as sess:
                    tf.initialize_all_variables().run()
                    eval_rmse = evaluate_rmse(train_dir, n_list, -1, batch_size, b_bound, dim, nn_approx, echo=True)
                    print("campaign={}\tfull eval rmse={}".format(camp, eval_rmse / avg_theta))


    def load_nn_approximator(self, input_type, model_path):
        if input_type == "txt":
            self.params = []
            with open(model_path, "r") as fin:
                line = fin.readline()
                line = line[:len(line) - 1].split("\t")
                self.net_type = line[0]
                if self.net_type == "nn":
                    depth = int(line[1])
                    h_dims = line[2].split("_")
                    for i in range(len(h_dims)):
                        h_dims[i] = int(h_dims[i])
                    act_func = line[3]
                    self.net_argv = [depth, h_dims, act_func]
                    self.dim = h_dims[0]
                    for i in range(depth - 1):
                        line = fin.readline()
                        line = line[:len(line) - 1].split("\t")
                        Wi = []
                        for item in line[1:]:
                            item = item.split("_")
                            item = str_list2float_list(item)
                            Wi.append(item)
                        line = fin.readline()
                        line = line[:len(line) - 1].split("\t")
                        bi = str_list2float_list(line[1:])
                        self.params.append((Wi, bi))
        elif input_type == "pickle":
            var_map = pickle.load(open(model_path, "rb"))
            self.net_type = var_map["net_type"]
            if self.net_type == "nn":
                depth = var_map["depth"]
                h_dims = var_map["h_dims"]
                act_func = var_map["act_func"]
                self.net_argv = [depth, h_dims, act_func]
                self.dim = h_dims[0]
                batch_size = 100
                self.nn_approx = NN_Approximator(self.net_type, self.net_argv, model_path, None, [self.dim],
                                                 batch_size, ['adam', 1e-4, 1e-8, 'mean'])
                self.sess = tf.Session(graph=self.nn_approx.graph)
                self.sess.run(self.nn_approx.init)

    def forward(self, x_vec):
        if self.sess:
            x_vec = np.array(x_vec).reshape(1, len(x_vec))
            feed_dict = {
                self.nn_approx.x_vec: x_vec
            }
            pred = self.sess.run(self.nn_approx.value_prediction, feed_dict=feed_dict)
            pred = pred.flatten()
            pred = pred[0]
        elif len(self.params) > 0:
            if self.net_type == "nn":
                depth, h_dims, act_func = self.net_argv
                z = x_vec
                for _i in range(depth - 2):
                    Wi, bi = self.params[_i]
                    a = [0] * h_dims[_i + 1]
                    for _j in range(h_dims[_i + 1]):
                        for _k in range(h_dims[_i]):
                            a[_j] += Wi[_j][_k] * z[_k]
                        a[_j] += bi[_j]
                    z = [0] * len(a)
                    for _j in range(h_dims[_i + 1]):
                        z[_j] = activate_calc(act_func, a[_j])
                W, b = self.params[depth - 2]
                pred = 0
                for _j in range(len(z)):
                    pred += W[0][_j] * z[_j]
                pred += b[0]
        return pred


    def get_Dnb(self, n, b):
        if n < len(self.D_info):
            if 0 <= self.D_info[n][1] <= b:
                return 0
            if b < len(self.D_point[n]):
                return self.D_point[n][b]
        x_vec = [n, b]
        if self.dim == 3:
            x_vec.append(b / n)
        dnb = self.forward(x_vec)
        dnb = max(dnb, 0)
        return dnb


    def bid(self, n, b, theta, max_bid):
        if n > self.N_bound:
            return self.bid(self.N_bound, int(b / n * self.N_bound), theta, max_bid)
        if b > self.B_bound:
            return self.bid(int(n / b * self.B_bound), self.B_bound, theta, max_bid)
        a = 0
        value = self.v1 * theta
        for delta in range(1, min(b, max_bid) + 1):
            dnb = self.get_Dnb(n - 1, b - delta)
            value -= self.gamma * dnb + self.v0
            if value >= 0:
                a = delta
            else:
                break
        return a


    def run(self, bid_log_path, N, c0, max_bid, save_log=False, bid_factor=1):

        auction = 0
        imp = 0
        clk = 0
        cost = 0

        if save_log:
            log_in = open(bid_log_path, "w")
        B = int(self.cpm * c0 * N)

        episode = 1
        n = N
        b = B

        click = 0
        theta, price = self.env.reset()

        done = False

        while not done:

            t0 = time.time()

            action = self.bid(n, b, theta, max_bid) * bid_factor
            #print("n={}, b={}, theta={}, max_bid={}, action={}, bid_factor={}".format(n, b, theta, max_bid, action, bid_factor))
            action = min(int(action), min(b, max_bid))

            t1 = time.time()
            log = str(t1 - t0) + "\t{}\t{}_{}\t{}_{}_{}\t{}_{}\t".format(
                episode, b, n, action, price, click, clk, imp)
            if save_log:
                log_in.write(log + "\n")

            done, new_theta, new_price, result_imp, result_click = self.env.step(action)
            #print("action = {}, price = {}".format( action, price))

            if result_imp > 0:
                imp += 1
                if result_click == 1:
                    clk += 1
                b -= price
                cost += price
            n -= 1
            auction += 1

            if n == 0:
                episode += 1
                n = N
                b = B

            theta = new_theta
            price = new_price

        if save_log:
            log_in.flush()
            log_in.close()

        return auction, imp, clk, cost



