from maml_rl.policies.continuous_mlp import ContinuousMLPPolicy
from maml_rl.envs.bidding import BiddingMDPEnv
from maml_rl.sampler import BatchSampler
import config
import os
import time
import math
from bidding_environment import BidEnv
from bidding_agent_rtb_rl_dp_tabular import bidding_agent_rtb_rl_dp_tabular
from utils import Opt_Obj, calc_m_pdf
from utils import load_data
from utils import write_log
from utils import getTime
from utils import evaluate_rmse
import numpy as np
import datetime as dt
from sklearn.metrics import mean_squared_error
import torch , torchvision

cuda = torch.device("cuda")
cpu = torch.device("cpu")


def evaluate_rmse_torch(train_dir, n_list, b_sample_size, batch_size, b_bound, dim, policy, echo=False):
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


            with torch.no_grad():
                batch_x = torch.FloatTensor(batch_x_vecs).to(cuda)
                action_distribution = policy(batch_x)
                batch_y_hat = action_distribution.sample()
                #batch_predictions = batch_y_hat.detach().to(cpu).numpy()
                batch_predictions = batch_y_hat.to(cpu).numpy()


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
    #
    # for _i in range(len(buf_x_vecs)):
    #     x_vec = buf_x_vecs[_i: (_i + 1)]
    #     value_label = buf_value_labels[_i]
    #     if value_label == 0:
    #         continue
    #
    #     with torch.no_grad():
    #         batch_x = torch.FloatTensor(x_vecs).to(cuda)
    #         batch_y_hat = policy(batch_x)
    #         pred = batch_y_hat.detach().to(cpu).numpy()
    #
    #     pred = pred.flatten()
    #     pred = pred[0]
    #     square_error += (value_label - pred) ** 2
    #     cnt += 1

    return np.sqrt(square_error / cnt)

def approximate(stop_after_first_it, policy , learning_rate, model, src, camp, N, D_function_path, large_storage_folder, NN_model_path, NN_model_txt_path, opt_obj, camp_info):
    seeds = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC,
             0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]

    if model == "dnb":
        dim = 2
        net_type = "nn"

        obj_type = "clk"
        clk_vp = 1
        tag = src + "_" + camp + "_" + model + "_" + net_type + "_N={}_{}".format(N, obj_type) + "_" + getTime()

        avg_theta = camp_info["clk_train"] / camp_info["imp_train"]
        if obj_type == "profit":
            avg_theta *= opt_obj.clk_v

        b_bound = 800
        n_bound = 50
        max_train_round = 10000
        #max_train_round = 2
        final_model_path = NN_model_path

        n_sample_size = 50
        #n_sample_size = 2
        b_sample_size = 200
        #b_sample_size = 2
        eval_n_sample_size = 500
        eval_b_sample_size = 1000
        batch_size = n_sample_size * b_sample_size

        net_argv = [4, [dim, 30, 15, 1], "tanh"]
        init_rag = avg_theta
        nn_approx = policy

        # need to split the D_function_path into one line files.
        train_dir = large_storage_folder + "/../fa-train/rlb_dnb_gamma=1_N={}_{}_1/".format(N, obj_type)
        # print("train_dir = {}".format(train_dir))
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

        # print("n_list = {}".format(n_list))

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

        # print(tag)
        # print(nn_approx.log)

        if mode == "train":
            #if save_model:
            #    write_log(log_path, nn_approx.log)


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
                # print("_round = {}".format(_round))
                for _i in range(_round):
                    batch_n = n_list[_i * n_sample_size: (_i + 1) * n_sample_size]
                    batch_x_vecs, batch_value_labels = load_data(train_dir, batch_n, b_sample_size, b_bound,
                                                                 dim)
                    # print("batch_x_vecs, batch_value_labels")
                    # print(batch_x_vecs, batch_value_labels)

                    batch_x = torch.FloatTensor(batch_x_vecs).to(cuda)
                    batch_y = torch.FloatTensor(batch_value_labels).to(cuda)

                    # print(batch_x.shape)
                    # print(batch_y.shape)
                    #
                    # print(batch_x)
                    # print(batch_y)
                    #
                    # print("batch_x, batch_y")

                    #batch_y_hat = policy(batch_x)
                    action_distribution = policy(batch_x)
                    batch_y_hat = action_distribution.sample()
                    #actions_tensor



                    # print("batch_y_hat")
                    # print(batch_y_hat)
                    # print(batch_y_hat.type)
                    # print(batch_y_hat.shape)

                    #reward = (batch_y_hat - batch_y).pow(2).sum() / batch_y_hat.shape[0]
                    reward = (batch_y_hat - batch_y).pow(2)
                    reward = -reward # The smaller the MSE the better.

                    loss = -(action_distribution.log_prob(batch_y_hat)) * reward
                    loss = loss.sum() / batch_y_hat.shape[0]

                    # print("loss")
                    # print(loss)
                    # print(loss.type)
                    # print(loss.shape)

                    # Zero gradients before running backward.
                    policy.zero_grad()

                    # Calculates gradients
                    loss.backward()

                    # Apply gradient descent
                    with torch.no_grad():
                        for param in policy.parameters():
                            param -= learning_rate * param.grad



                    #_, loss, batch_predictions = 1,2,3
                    #batch_predictions = batch_y_hat.detach().to(cpu).numpy()
                    batch_predictions = batch_y_hat.to(cpu).numpy()
                    loss = loss.to(cpu).item()

                    # print(batch_predictions)
                    #
                    # if np.isnan(loss).any():
                    #     print(loss)
                    #     exit()

                    #buf_loss.append(np.sqrt(loss) / avg_theta)
                    buf_loss.append(loss)
                    buf_labels.extend(batch_value_labels.astype(np.float64).flatten())
                    buf_predictions.extend(batch_predictions.astype(np.float64).flatten())

                buf_loss = np.array(buf_loss)
                # if np.isnan(buf_loss).any():
                #     print(buf_loss)
                #     exit()
                buf_rmse = np.sqrt(mean_squared_error(buf_labels, buf_predictions))
                buf_log = "buf loss, max={:.6f}\tmin={:.6f}\tmean={:.6f}\tbuf rmse={}\ttime={}".format(
                    buf_loss.max(), buf_loss.min(), buf_loss.mean(), buf_rmse / avg_theta, getTime())
                print(buf_log)

                np.random.shuffle(n_list)

                if stop_after_first_it:
                    return

                eval_rmse = evaluate_rmse_torch(train_dir, n_list[:eval_n_sample_size], eval_b_sample_size,
                                        batch_size,
                                        b_bound, dim, policy)
                eval_log = "iteration={}\ttime={}\teval rmse={}\tbuf rmse={}" \
                  .format(_iter, time.time() - start_time, eval_rmse / avg_theta, buf_rmse / avg_theta)
                print(eval_log)

                #EVAL PHASE
                # eval_rmse = evaluate_rmse(train_dir, n_list[:eval_n_sample_size], eval_b_sample_size,
                #                           batch_size,
                #                           b_bound, dim, nn_approx)
                # eval_log = "iteration={}\ttime={}\teval rmse={}\tbuf rmse={}" \
                #     .format(_iter, time.time() - start_time, eval_rmse / avg_theta, buf_rmse / avg_theta)
                # print(eval_log)
                #EVAL PHASE

                # if save_model:
                #     write_log(log_path, eval_log)
                #     nn_approx.dump(model_path + "{}_{}.pickle".format(tag, _iter), net_type, net_argv)
                #     nn_approx.pickle2txt(model_path + "{}_{}.pickle".format(tag, _iter),
                #                          model_path + "{}_{}.txt".format(tag, _iter))
                #     n_perf = (buf_rmse + eval_rmse) / avg_theta
                #     if n_perf < perf:
                #         perf = n_perf
                #         nn_approx.dump(final_model_path, net_type, net_argv)
                start_time = time.time()
                if _iter >= max_train_round:
                    torch.save(nn_approx.state_dict(), final_model_path)
                    break

        elif mode == "eval":
            print("eval")
            # with tf.Session(graph=nn_approx.graph) as sess:
            #     tf.initialize_all_variables().run()
            #     eval_rmse = evaluate_rmse(train_dir, n_list, -1, batch_size, b_bound, dim, nn_approx, echo=True)
            #     print("campaign={}\tfull eval rmse={}".format(camp, eval_rmse / avg_theta))

def create_D_function(camp):
    large_storage_media = "/media/onetbssd/rlb/"

    src = "ipinyou"

    obj_type = "clk"
    clk_vp = 1
    # N = 10000
    N = 500
    c0 = 1 / 8
    gamma = 1
    overwrite = False

    if src == "ipinyou":
        camps = config.ipinyou_camps_to_test
        data_path = config.ipinyouPath
        max_market_price = config.ipinyou_max_market_price

    camp_info = config.get_camp_info(camp, src)
    aution_in_file = data_path + camp + "/test.theta.txt"
    opt_obj = Opt_Obj(obj_type, int(clk_vp * camp_info["cost_train"] / camp_info["clk_train"]))
    B = int(camp_info["cost_train"] / camp_info["imp_train"] * c0 * N)

    large_storage_folder = large_storage_media + src + "/" + camp + "/bid-model/"

    if not os.path.exists(large_storage_folder):
        os.makedirs(large_storage_folder)

    env = BidEnv(camp_info, aution_in_file)
    agent = bidding_agent_rtb_rl_dp_tabular()
    agent.init(env, camp_info, opt_obj, gamma)

    setting = "{}, camp={}, algo={}, N={}, c0={}" \
        .format(src, camp, "rlb_rl_fa", N, c0)
    bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

    # Approximating D(t,b) function
    m_pdf = calc_m_pdf(camp_info["price_counter_train"])
    D_function_path = large_storage_folder + "rlb_dnb_gamma={}_N={}_{}.txt".format(gamma, N, obj_type)
    print("D_function_path = " + D_function_path)
    if (not os.path.isfile(D_function_path)) or overwrite:
        # print("START: Approximating V function by dynamic programming... ")
        agent.calc_Dnb(N, B, max_market_price, m_pdf, D_function_path)
        # print("END: Approximating V function by dynamic programming.")

    # Then train a NN using the Dnd function
    NN_model_path = large_storage_folder + "fa_dnb_gamma={}_N={}_{}.pickle".format(gamma, N, obj_type)
    NN_model_txt_path = large_storage_folder + "fa_dnb_gamma={}_N={}_{}.txt".format(gamma, N, obj_type)

    return agent, src, N, D_function_path, large_storage_folder, NN_model_path, NN_model_txt_path, opt_obj, camp_info
    # END -----------



# START ----------
def main():

    sampler = BatchSampler('BiddingMDP-v0', batch_size=50,
                                   num_workers=2)

    policy = ContinuousMLPPolicy(
                    int(np.prod(sampler.envs.observation_space.shape)),
                    int(np.prod(sampler.envs.action_space.shape)),
                    hidden_sizes=(1000,) * 3)



    policy.cuda()


    print(str(dt.datetime.now()) + " - policy created")

    # Create D function
    camp = "2997"
    agent, src, N, D_function_path, large_storage_folder, NN_model_path, NN_model_txt_path, opt_obj, camp_info = create_D_function(camp)

    print(str(dt.datetime.now()) + " - D function created")

    # Approximate d function:
    model = "dnb"
    learning_rate = 1e-4
    stop_after_first_it = False
    approximate(stop_after_first_it, policy, learning_rate, model, src, camp, N, D_function_path, large_storage_folder, NN_model_path, NN_model_txt_path, opt_obj, camp_info)



    policy2 = ContinuousMLPPolicy(
                    int(np.prod(sampler.envs.observation_space.shape)),
                    int(np.prod(sampler.envs.action_space.shape)),
                    hidden_sizes=(1000,) * 3)

    policy2.load_state_dict(torch.load(NN_model_path))
    policy2.cuda()
    stop_after_first_it = True
    approximate(stop_after_first_it, policy2, learning_rate, model, src, camp, N, D_function_path, large_storage_folder, NN_model_path, NN_model_txt_path, opt_obj, camp_info)

#main()