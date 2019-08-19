import config
import sys
import os
import numpy as np
import pickle
import datetime as dt
from utils import Opt_Obj
from utils import calc_m_pdf
from bidding_environment import BidEnv
from bidding_agent_linear import bidding_agent_linear
from bidding_agent_rtb_rl_dp_tabular import bidding_agent_rtb_rl_dp_tabular
from bidding_agent_rtb_rl_fa import bidding_agent_rtb_rl_fa
from bidding_agent_meta import bidding_agent_meta
from utils import getTime
from maml_rl.sampler import BatchSampler
from maml_rl.policies.continuous_mlp import ContinuousMLPPolicy
from bid_train_nn_with_d_function import approximate
from bid_train_nn_with_d_function import create_D_function
import torch
import numpy as np

obj_type = "clk"
clk_vp = 1
#N = 10000
N = 1000
c0 = 1 / 8
gamma = 1

#agents_to_execute = ['lin', 'rlb_dp_tabular']
#agents_to_execute = ['lin','rlb_rl_dp_tabular', "rlb_rl_fa", 'meta']
#agents_to_execute = ['meta']
#agents_to_execute = ['lin','meta_imitation']
agents_to_execute = ['meta_imitation', 'lin', 'rlb_rl_dp_tabular']
#agents_to_execute = ['lin', 'rlb_rl_dp_tabular','meta']
#agents_to_execute = ["rlb_rl_fa"]

src = "ipinyou"

log_in = open(config.projectPath + "bid-performance/{}_N={}_c0={}_obj={}_clkvp={}.txt"
              .format(src, N, c0, obj_type, clk_vp), "w")
print("logs in {}".format(log_in.name))

log = "{:<55}\t{:>10}\t{:>8}\t{:>10}\t{:>8}\t{:>8}\t{:>9}\t{:>8}\t{:>8}" \
            .format("setting", "objective", "auction", "impression", "click", "cost", "win-rate", "CPM", "eCPC")
print(log)
log_in.write(log + "\n")

if src == "ipinyou":
    camps = config.ipinyou_camps_to_test
    data_path = config.ipinyouPath
    max_market_price = config.ipinyou_max_market_price



for camp in camps:
    camp_info = config.get_camp_info(camp, src)
    aution_in_file = data_path + camp + "/test.theta.txt"
    opt_obj = Opt_Obj(obj_type, int(clk_vp * camp_info["cost_train"] / camp_info["clk_train"]))
    B = int(camp_info["cost_train"] / camp_info["imp_train"] * c0 * N)

    large_storage_media = "/media/onetbssd/rlb/"

    # Create log file folder location.
    if not os.path.exists(config.projectPath + "bid-log"):
        os.makedirs(config.projectPath + "bid-log")

    # Linear-Bid
    if 'lin' in agents_to_execute:

        train_camps = []

        for train_camp in config.ipinyou_camps:
            if train_camp != camp:
                train_camps.append(train_camp)

        result_array = []
        for train_camp in train_camps:

            train_aution_in_file = data_path + train_camp + "/test.theta.txt"
            train_camp_info = config.get_camp_info(train_camp, src)
            train_opt_obj = Opt_Obj(obj_type, int(clk_vp * train_camp_info["cost_train"] / train_camp_info["clk_train"]))
            train_env = BidEnv(train_camp_info, train_aution_in_file)
            training_agent = bidding_agent_linear()
            training_agent.init(train_env, train_camp_info)
            setting = "{}, camp={}, algo={}, N={}, c0={}" \
                .format(src, camp, "lin_bid", N, c0)
            bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

            if not os.path.exists(data_path + camp + "/train_camp/" + train_camp + "/bid-model/"):
                os.makedirs(data_path + camp + "/train_camp/" + train_camp + "/bid-model/")

            model_path = data_path + camp + "/train_camp/" + train_camp + "/bid-model/{}_{}_{}_{}_{}.pickle".format("lin-bid", N, c0, obj_type, opt_obj.clk_v)
            training_agent.parameter_tune(train_opt_obj,  model_path, N, c0, max_market_price,
                                   max_market_price, load=False)

            testing_env = BidEnv(camp_info, aution_in_file)
            testing_agent = bidding_agent_linear()
            testing_agent.init(testing_env, camp_info)
            testing_agent.parameter_tune(opt_obj, model_path, N, c0, max_market_price,
                                          max_market_price, load=False)


            (auction, imp, clk, cost) = testing_agent.run(bid_log_path, N, c0,
                                                    max_market_price,  save_log=False)


            result_array.append(np.array([[auction,imp,clk,cost]]))

        result = np.concatenate(result_array, axis=0)
        result = np.apply_along_axis(np.average, 0, result)

        auction = result[0]
        imp = result[1]
        clk = result[2]
        cost = result[3]

        win_rate = imp / auction * 100
        cpm = (cost / 1000) / imp * 1000
        ecpc = (cost / 1000) / clk
        obj = opt_obj.get_obj(imp, clk, cost)
        log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
            .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
        print(log)
        log_in.write(log + "\n")


    # RLB-DP-TABULAR
    if 'rlb_rl_dp_tabular' in agents_to_execute:

        train_camps = []

        for train_camp in config.ipinyou_camps:
            if train_camp != camp:
                train_camps.append(train_camp)

        result_array = []
        for train_camp in train_camps:
            train_aution_in_file = data_path + train_camp + "/test.theta.txt"
            train_camp_info = config.get_camp_info(train_camp, src)
            train_opt_obj = Opt_Obj(obj_type, int(clk_vp * train_camp_info["cost_train"] / train_camp_info["clk_train"]))
            train_B = int(train_camp_info["cost_train"] / train_camp_info["imp_train"] * c0 * N)
            train_env = BidEnv(train_camp_info, train_aution_in_file)

            train_agent = bidding_agent_rtb_rl_dp_tabular()
            train_agent.init(train_env, train_camp_info, train_opt_obj, gamma)

            setting = "{}, camp={}, algo={}, N={}, c0={}" \
                .format(src, camp, "rlb_rl_dp_tabular", N, c0)
            bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

            # Approximating V function
            #print("START: Approximating V function by dynamic programming... ")
            train_m_pdf = calc_m_pdf(train_camp_info["price_counter_train"])

            if not os.path.exists(data_path + camp + "/train_camp/" + train_camp + "/bid-model"):
                os.makedirs(data_path + camp + "/train_camp/" + train_camp + "/bid-model")

            model_path = data_path + camp + "/train_camp/" + train_camp + "/bid-model/v_nb_N={}.txt".format(N)
            train_agent.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, train_m_pdf, model_path)
            #print("END: Approximating V function by dynamic programming.")


            testing_env = BidEnv(camp_info, aution_in_file)
            testing_agent = bidding_agent_rtb_rl_dp_tabular()
            testing_agent.init(testing_env, camp_info, opt_obj, gamma)
            testing_m_pdf = calc_m_pdf(camp_info["price_counter_train"])

            # Load function
            testing_agent.load_value_function(N, B, model_path)

            #print("START: Run real time bidding with reinforcement learning, tabular")
            (auction, imp, clk, cost) = testing_agent.run(bid_log_path, N, c0,
                                                     max_market_price, delimiter=" ", save_log=False)
            #print("END: Run real time bidding with reinforcement learning, tabular")

            result_array.append(np.array([[auction,imp,clk,cost]]))

        result = np.concatenate(result_array, axis=0)
        result = np.apply_along_axis(np.average, 0, result)

        auction = result[0]
        imp = result[1]
        clk = result[2]
        cost = result[3]


        win_rate = imp / auction * 100
        cpm = (cost / 1000) / imp * 1000
        ecpc = (cost / 1000) / clk
        obj = opt_obj.get_obj(imp, clk, cost)
        log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
            .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
        print(log)
        log_in.write(log + "\n")

    # RLB-FA: real time D(t,b) function approximation.
    if 'rlb_rl_fa' in agents_to_execute:

        overwrite = False
        # First train DP


        train_camps = []

        for train_camp in config.ipinyou_camps:
            if train_camp != camp:
                train_camps.append(train_camp)

        result_array = []
        for train_camp in train_camps:

            # large_storage_media = "."
            large_storage_folder = large_storage_media + src + "/" + camp + "/train_camp/" + train_camp + "/bid-model/"

            if not os.path.exists(large_storage_folder):
                os.makedirs(large_storage_folder)

            train_aution_in_file = data_path + train_camp + "/test.theta.txt"
            train_camp_info = config.get_camp_info(train_camp, src)
            train_opt_obj = Opt_Obj(obj_type,
                                    int(clk_vp * train_camp_info["cost_train"] / train_camp_info["clk_train"]))
            train_B = int(train_camp_info["cost_train"] / train_camp_info["imp_train"] * c0 * N)

            train_env = BidEnv(train_camp_info, train_aution_in_file)
            train_agent = bidding_agent_rtb_rl_dp_tabular()
            train_agent.init(train_env, train_camp_info, train_opt_obj, gamma)

            setting = "{}, camp={}, algo={}, N={}, c0={}" \
                .format(src, camp, "rlb_rl_fa", N, c0)
            bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

            # Approximating D(t,b) function
            train_m_pdf = calc_m_pdf(train_camp_info["price_counter_train"])
            D_function_path = large_storage_folder + "rlb_dnb_gamma={}_N={}_{}.txt".format(gamma, N, obj_type)
            if (not os.path.isfile(D_function_path)) or overwrite:
                # print("START: Approximating V function by dynamic programming... ")
                train_agent.calc_Dnb(N, train_B, max_market_price, train_m_pdf, D_function_path)
                # print("END: Approximating V function by dynamic programming.")


            # Then train a NN using the Dnd function
            NN_model_path = large_storage_folder + "fa_dnb_gamma={}_N={}_{}.pickle".format(gamma, N, obj_type)
            NN_model_txt_path = large_storage_folder + "fa_dnb_gamma={}_N={}_{}.txt".format(gamma, N, obj_type)
            if (not os.path.isfile(NN_model_path)) or overwrite:
                train_agent = bidding_agent_rtb_rl_fa()
                train_agent.init(env, camp_info, gamma, opt_obj)
                train_agent.approximate("dnb", src, train_camp, N, D_function_path, large_storage_folder, NN_model_path, NN_model_txt_path)


            # Then use it
            testing_env = BidEnv(camp_info, aution_in_file)
            testing_agent = bidding_agent_rtb_rl_fa()
            testing_agent.init(testing_env, camp_info, gamma, opt_obj)

            testing_agent.load_nn_approximator("pickle", NN_model_path)


            bid_factor_file = large_storage_folder + "{}_{}_{}_{}_{}.pickle".format("rlb_bid_factor", N, c0,
                                                                                           obj_type,
                                                                                           opt_obj.clk_v)
            if os.path.isfile(bid_factor_file):
                bf = pickle.load(open(bid_factor_file, "rb"))["bid-factor"]
            else:
                bf = 1

            (auction, imp, clk, cost) = testing_agent.run(bid_log_path, N, c0,
                                                      max_market_price, save_log=True, bid_factor=bf)

            result_array.append(np.array([[auction, imp, clk, cost]]))

        result = np.concatenate(result_array, axis=0)
        result = np.apply_along_axis(np.average, 0, result)

        auction = result[0]
        imp = result[1]
        clk = result[2]
        cost = result[3]

        win_rate = imp / auction * 100
        cpm = (cost / 1000) / (imp + 1.0e-14) * 1000
        ecpc = (cost / 1000) / (clk + 1.0e-14)
        obj = opt_obj.get_obj(imp, clk, cost)
        log = "{:<80}\t{:>10}\t{:>8}\t{:>10}\t{:>8}\t{:>8}\t{:>8.2f}%\t{:>8.2f}\t{:>8.2f}" \
            .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
        print(log)


        log_in.write(log + "\n")

        #exit()
    if 'meta' in agents_to_execute:

        # Set the camp to train on:
        config.ipinyou_camps_target = camp

        # Runs meta RL and stores final model.
        agent = bidding_agent_meta(camp_info)
        large_storage_folder = large_storage_media + src + "/" + camp + "/bid-model/"
        print(getTime() + ":BEGIN meta training")
        NN_model_path = agent.run_meta_training(large_storage_folder)
        print(getTime() + ":END meta training")

        # Read final model and evaluate.
        agent.load_model(NN_model_path)

        # prepare to run traditional bidding on the meta-trained model.
        setting = "{}, camp={}, algo={}, N={}, c0={}" \
            .format(src, camp, "meta_bid", N, c0)
        bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)
        env = BidEnv(camp_info, aution_in_file)

        (auction, imp, clk, cost) = agent.run(env, bid_log_path, N, c0,
                                              max_market_price, save_log=True)

        win_rate = imp / auction * 100
        cpm = (cost / 1000) / (imp + 1.0e-14) * 1000
        ecpc = (cost / 1000) / (clk + 1.0e-14)
        obj = opt_obj.get_obj(imp, clk, cost)
        log = "{:<80}\t{:>10}\t{:>8}\t{:>10}\t{:>8}\t{:>8}\t{:>8.2f}%\t{:>8.2f}\t{:>8.2f}" \
            .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
        print(log)

    if 'meta_imitation' in agents_to_execute:
        train_camps = []
        overwrite = False
        actual_camp = camp
        actual_camp_info = camp_info
        actual_aution_in_file = data_path + camp + "/test.theta.txt"
        actual_N = N

        for train_camp in config.ipinyou_camps:
            if train_camp != camp:
                train_camps.append(train_camp)



        sampler = BatchSampler('BiddingMDP-v0', batch_size=50,
                               num_workers=2)

        policy = ContinuousMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(400,) * 3)

        policy.cuda()

        print(str(dt.datetime.now()) + " meta imitiation init " + " - policy created ")

        # Create D function
        camp = train_camps[0]
        camp_info = config.get_camp_info(camp, src)
        aution_in_file = data_path + camp + "/test.theta.txt"

        #if (not os.path.exists(D_function_path)) or overwrite:
        agent, src, N, D_function_path, large_storage_folder, NN_model_path, NN_model_txt_path, opt_obj, camp_info, X, Y = create_D_function(camp, max_market_price)
        print("Model path " + NN_model_path)

        print(str(dt.datetime.now()) + " meta imitiation init " + " - D function created ")



        # Approximate d function:
        model = "dnb"
        learning_rate = 1e-4
        stop_after_first_it = False

        # print("(not os.path.exists(NN_model_path)) or overwrite")
        # print((not os.path.exists(NN_model_path)) or overwrite)
        # exit()

        if (not os.path.exists(NN_model_path)) or overwrite:
            print("NN_model_path does not exist")
            print(NN_model_path)
            approximate(stop_after_first_it, policy, learning_rate, model, src, camp, N,
                        large_storage_folder, NN_model_path, NN_model_txt_path, opt_obj, camp_info, X, Y)

        print(str(dt.datetime.now()) + " meta imitiation init " + " - finish training model ")

        # Load the nn weights:
        imitation_policy = ContinuousMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(400,) * 3)

        imitation_policy.load_state_dict(torch.load(NN_model_path))
        imitation_policy.cuda()

        print(str(dt.datetime.now()) + " meta imitiation init " + " - finish saving/loading model ")

        # Set the camp to train on:
        config.ipinyou_camps_target = camp

        # Runs meta RL and stores final model.
        agent = bidding_agent_meta(camp_info)
        torch.cuda.empty_cache()
        large_storage_folder = large_storage_media + src + "/" + camp + "/bid-model/"
        print(getTime() + ":BEGIN meta training")
    print(getTime() + ":END meta training")



        # Read final model and evaluate.
        agent.load_model(NN_model_path_final)
        #agent.load_model(NN_model_path)

        # prepare to run traditional bidding on the meta-trained model.
        setting = "{}, camp={}, algo={}, N={}, c0={}" \
            .format(src, actual_camp, "meta_bid", actual_N, c0)
        bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

        env = BidEnv(actual_camp_info, actual_aution_in_file)

        (auction, imp, clk, cost) = agent.run(env, bid_log_path, actual_N, c0,
                                              max_market_price, save_log=True)

        win_rate = imp / auction * 100
        cpm = (cost / 1000) / (imp + 1.0e-14) * 1000
        ecpc = (cost / 1000) / (clk + 1.0e-14)
        obj = opt_obj.get_obj(imp, clk, cost)
        log = "{:<80}\t{:>10}\t{:>8}\t{:>10}\t{:>8}\t{:>8}\t{:>8.2f}%\t{:>8.2f}\t{:>8.2f}" \
            .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
        print(log)

    print( " Finish processing camp = {}".format(camp))

print("Finish processing all camps")



