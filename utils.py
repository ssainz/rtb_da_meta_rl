import numpy as np
import time
import tensorflow as tf
import pickle
import math
import os


# obj_type: clk, profit, imp
class Opt_Obj:
    def __init__(self, obj_type="clk", clk_v=500):
        self.obj_type = obj_type
        self.clk_v = clk_v
        if obj_type == "clk":
            self.v1 = 1
            self.v0 = 0
            self.w = 0
        elif obj_type == "profit":
            self.v1 = clk_v
            self.v0 = 1
            self.w = 0
        else:
            self.v1 = 0
            self.v0 = 0
            self.w = 1

    def get_obj(self, imp, clk, cost):
        return self.v1 * clk - self.v0 * cost + self.w * imp


def write_log(log_path, line, echo=False):
    with open(log_path, "a") as log_in:
        log_in.write(line + "\n")
        if echo:
            print(line)

def getTime():
	return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def calc_m_pdf(m_counter, laplace=1):
    m_pdf = [0] * len(m_counter)
    sum = 0
    for i in range(0, len(m_counter)):
        sum += m_counter[i]
    for i in range(0, len(m_counter)):
        m_pdf[i] = (m_counter[i] + laplace) / (
            sum + len(m_counter) * laplace)
    return m_pdf


def load_data(train_dir, batch_n, b_sample_size, b_bound, dim):
    NB = []
    Dnb = []
    for n in batch_n:
        with open(train_dir + "{}.txt".format(n)) as fin:
            line = fin.readline()
            line = line[:len(line) - 1].split("\t")
            line = line[1:]
            b_list = [i for i in range(b_bound, len(line))]
            np.random.shuffle(b_list)
            if b_sample_size > 0:
                b_list = b_list[:b_sample_size]

            for b in b_list:
                nb = [n, b]
                if dim == 3:
                    nb.append(b / n)
                dnb = float(line[b])
                NB.append(nb)
                Dnb.append([dnb])
    NB = np.array(NB)
    Dnb = np.array(Dnb)
    return NB, Dnb


def evaluate_rmse(train_dir, n_list, b_sample_size, batch_size, b_bound, dim, model, echo=False):
    preds = []
    labels = []

    square_error = 0
    cnt = 0
    buf_x_vecs = []
    buf_value_labels = []
    for n in n_list:
        x_vecs, value_labels = load_data(train_dir, [n], b_sample_size, b_bound, dim)

        buf_x_vecs.extend(x_vecs)
        buf_value_labels.extend(value_labels.flatten())
        while len(buf_x_vecs) >= batch_size:
            batch_x_vecs = buf_x_vecs[0: batch_size]
            batch_value_labels = buf_value_labels[0: batch_size]
            feed_dict = {
                model.batch_x_vecs: batch_x_vecs
            }
            batch_predictions = model.batch_value_predictions.eval(feed_dict=feed_dict)
            batch_predictions = batch_predictions.flatten().tolist()
            for _i in range(batch_size):
                if batch_value_labels[_i] == 0:
                    continue
                square_error += (batch_value_labels[_i] - batch_predictions[_i]) ** 2
                cnt += 1
            buf_x_vecs = buf_x_vecs[batch_size:]
            buf_value_labels = buf_value_labels[batch_size:]
        if echo:
            print("{}\t{}\t{}".format(n, np.sqrt(square_error / cnt), getTime()))

    for _i in range(len(buf_x_vecs)):
        x_vec = buf_x_vecs[_i: (_i + 1)]
        value_label = buf_value_labels[_i]
        if value_label == 0:
            continue
        feed_dict = {
            model.x_vec: x_vec
        }
        pred = model.value_prediction.eval(feed_dict=feed_dict)
        pred = pred.flatten()
        pred = pred[0]
        square_error += (value_label - pred) ** 2
        cnt += 1

    return np.sqrt(square_error / cnt)




def activate(act_func, x):
    if act_func == 'tanh':
        return tf.tanh(x)
    elif act_func == 'relu':
        return tf.nn.relu(x)
    else:
        return tf.sigmoid(x)

def init_var_map(init_path, _vars):
    if init_path:
        var_map = pickle.load(open(init_path, "rb"))
    else:
        var_map = {}

    for i in range(len(_vars)):
        key, shape, init_method, init_argv = _vars[i]
        if key not in var_map.keys():
            if init_method == "normal":
                mean, dev, seed = init_argv
                var_map[key] = tf.random_normal(shape, mean, dev, seed=seed)
            elif init_method == "uniform":
                min_val, max_val, seed = init_argv
                var_map[key] = tf.random_uniform(shape, min_val, max_val, seed=seed)
            else:
                var_map[key] = tf.zeros(shape)

    return var_map

def build_optimizer(opt_argv, loss):
    opt_method = opt_argv[0]
    if opt_method == 'adam':
        _learning_rate, _epsilon = opt_argv[1:3]
        opt = tf.train.AdamOptimizer(learning_rate=_learning_rate, epsilon=_epsilon).minimize(loss)
    elif opt_method == 'ftrl':
        _learning_rate = opt_argv[1]
        opt = tf.train.FtrlOptimizer(learning_rate=_learning_rate).minimize(loss)
    else:
        _learning_rate = opt_argv[1]
        opt = tf.train.GradientDescentOptimizer(learning_rate=_learning_rate).minimize(loss)
    return opt


def str_list2float_list(str_list):
    res = []
    for _str in str_list:
        res.append(float(_str))
    return res

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def activate_calc(act_func, x):
    if act_func == "tanh":
        return np.tanh(x)
    elif act_func == "relu":
        return max(0, x)
    else:
        return sigmoid(x)

def merge_files(files_list, training_file, k_shoots):

    memory = []
    for filename in files_list:
        with open(filename) as f:
            for line in f:
                memory.append(line)

    with open(training_file) as f:
        count = 0
        for line in f:
            memory.append(line)
            count += 1
            if count > k_shoots:
                break

    folder = generate_temp_folder()
    output_filename = folder + "merged_file_" + str(time.time()) + ".txt"
    with open(output_filename, 'w') as filehandle:
        filehandle.writelines("%s" % place for place in memory)

    return output_filename

def generate_temp_folder():

    current_dir = os.getcwd()
    final_dir = current_dir + "/merged_files/" + str(time.time()) + "/"
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)

    return final_dir