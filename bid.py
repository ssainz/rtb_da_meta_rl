import config
import sys
import os
import pickle
from utils import Opt_Obj
from utils import calc_m_pdf
from bidding_environment import BidEnv
from bidding_agent_linear import bidding_agent_linear
from bidding_agent_rtb_rl_dp_tabular import bidding_agent_rtb_rl_dp_tabular
from bidding_agent_rtb_rl_fa import bidding_agent_rtb_rl_fa

obj_type = "clk"
clk_vp = 1
N = 200
c0 = 1 / 8
gamma = 1

#agents_to_execute = ['lin', 'rlb_dp_tabular']
#agents_to_execute = ['lin','rlb_rl_dp_tabular', "rlb_rl_fa"]
agents_to_execute = ["rlb_rl_fa"]

src = "ipinyou"

log_in = open(config.projectPath + "bid-performance/{}_N={}_c0={}_obj={}_clkvp={}.txt"
              .format(src, N, c0, obj_type, clk_vp), "w")
print("logs in {}".format(log_in.name))

log = "{:<55}\t{:>10}\t{:>8}\t{:>10}\t{:>8}\t{:>8}\t{:>9}\t{:>8}\t{:>8}" \
            .format("setting", "objective", "auction", "impression", "click", "cost", "win-rate", "CPM", "eCPC")
print(log)
log_in.write(log + "\n")

if src == "ipinyou":
    camps = config.ipinyou_camps
    data_path = config.ipinyouPath
    max_market_price = config.ipinyou_max_market_price



for camp in camps:
    camp_info = config.get_camp_info(camp, src)
    aution_in_file = data_path + camp + "/test.theta.txt"
    opt_obj = Opt_Obj(obj_type, int(clk_vp * camp_info["cost_train"] / camp_info["clk_train"]))
    B = int(camp_info["cost_train"] / camp_info["imp_train"] * c0 * N)

    # Linear-Bid
    if 'lin' in agents_to_execute:

        env = BidEnv(camp_info, aution_in_file)

        agent = bidding_agent_linear()
        agent.init(env, camp_info)
        setting = "{}, camp={}, algo={}, N={}, c0={}" \
            .format(src, camp, "lin_bid", N, c0)
        bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)
        model_path = data_path + camp + "/bid-model/{}_{}_{}_{}_{}.pickle".format("lin-bid", N, c0, obj_type, opt_obj.clk_v)
        agent.parameter_tune(opt_obj,  model_path, N, c0, max_market_price,
                               max_market_price, load=False)
        (auction, imp, clk, cost) = agent.run(bid_log_path, N, c0,
                                                max_market_price,  save_log=False)

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


        env = BidEnv(camp_info, aution_in_file)

        agent = bidding_agent_rtb_rl_dp_tabular()
        agent.init(env, camp_info, opt_obj, gamma)

        setting = "{}, camp={}, algo={}, N={}, c0={}" \
            .format(src, camp, "rlb_rl_dp_tabular", N, c0)
        bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

        # Approximating V function
        #print("START: Approximating V function by dynamic programming... ")
        m_pdf = calc_m_pdf(camp_info["price_counter_train"])
        model_path = data_path + camp + "/bid-model/v_nb_N={}.txt".format(N)
        agent.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf, model_path)
        #print("END: Approximating V function by dynamic programming.")

        # Load function
        agent.load_value_function(N, B, model_path)

        #print("START: Run real time bidding with reinforcement learning, tabular")
        (auction, imp, clk, cost) = agent.run(bid_log_path, N, c0,
                                                 max_market_price, delimiter=" ", save_log=False)
        #print("END: Run real time bidding with reinforcement learning, tabular")

        win_rate = imp / auction * 100
        cpm = (cost / 1000) / imp * 1000
        ecpc = (cost / 1000) / clk
        obj = opt_obj.get_obj(imp, clk, cost)
        log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
            .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
        print(log)
        log_in.write(log + "\n")

    # RLB-FA: real time bidding tabular function approximation.
    if 'rlb_rl_fa' in agents_to_execute:

        overwrite = True
        # First train DP
        large_storage_folder = "/media/onetbssd/rlb/" + src + "/" + camp + "/bid-model/"

        if not os.path.exists(large_storage_folder):
            os.makedirs(large_storage_folder)

        env = BidEnv(camp_info, aution_in_file)
        agent = bidding_agent_rtb_rl_dp_tabular()
        agent.init(env, camp_info, opt_obj, gamma)

        setting = "{}, camp={}, algo={}, N={}, c0={}" \
            .format(src, camp, "rlb_rl_fa", N, c0)
        bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

        # Approximating D function
        m_pdf = calc_m_pdf(camp_info["price_counter_train"])
        D_function_path = large_storage_folder + "rlb_dnb_gamma={}_N={}_{}.txt".format(gamma, N, obj_type)
        if (not os.path.isfile(D_function_path)) or overwrite:
            # print("START: Approximating V function by dynamic programming... ")
            agent.calc_Dnb(N, B, max_market_price, m_pdf, D_function_path)
            # print("END: Approximating V function by dynamic programming.")


        # Then train a NN using the Dnd function
        NN_model_path = large_storage_folder + "fa_dnb_gamma={}_N={}_{}.pickle".format(gamma, N, obj_type)
        if (not os.path.isfile(NN_model_path)) or overwrite:
            agent = bidding_agent_rtb_rl_fa()
            agent.init(env, camp_info, gamma, opt_obj)
            agent.approximate("dnb", src, camp, N, D_function_path, large_storage_folder, NN_model_path)


        # Then use it
        env = BidEnv(camp_info, aution_in_file)
        agent = bidding_agent_rtb_rl_fa()
        agent.init(env, camp_info, gamma, opt_obj)

        agent.load_nn_approximator("pickle", NN_model_path)

        bid_factor_file = large_storage_folder + "{}_{}_{}_{}_{}.pickle".format("rlb_bid_factor", N, c0,
                                                                                       obj_type,
                                                                                       opt_obj.clk_v)
        if os.path.isfile(bid_factor_file):
            bf = pickle.load(open(bid_factor_file, "rb"))["bid-factor"]
        else:
            bf = 1

        (auction, imp, clk, cost) = agent.run(bid_log_path, N, c0,
                                                  max_market_price, save_log=True, bid_factor=bf)

        win_rate = imp / auction * 100
        cpm = (cost / 1000) / (imp + 1.0e-14) * 1000
        ecpc = (cost / 1000) / (clk + 1.0e-14)
        obj = opt_obj.get_obj(imp, clk, cost)
        log = "{:<80}\t{:>10}\t{:>8}\t{:>10}\t{:>8}\t{:>8}\t{:>8.2f}%\t{:>8.2f}\t{:>8.2f}" \
            .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
        print(log)
        log_in.write(log + "\n")


    # agent = bidding_agent_rtb_rl(camp_info, opt_obj, gamma)
    # setting = "{}, camp={}, algo={}, N={}, c0={}, obj={}, clk_v={}" \
    #     .format(src, camp, "rlb_dp_fa", N, c0, obj_type, opt_obj.clk_v)
    # bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)
    #
    # save_point_path = data_path + camp + "/bid-model/rlb_dnb_save_points_{}.txt".format(obj_type)
    # agent.load_save_points(save_point_path)
    #
    # # model_path = data_path + camp + "/bid-model/fa_dnb_{}.pickle".format(obj_type)
    # # rlb_dp_fa.load_nn_approximator("pickle", model_path)
    #
    # model_path = data_path + camp + "/bid-model/fa_dnb_{}.txt".format(obj_type)
    # agent.load_nn_approximator("txt", model_path)
    #
    # bid_factor_file = data_path + camp + "/bid-model/{}_{}_{}_{}_{}.pickle".format("rlb_bid_factor", N, c0, obj_type,
    #                                                                                opt_obj.clk_v)
    # if os.path.isfile(bid_factor_file):
    #     bf = pickle.load(open(bid_factor_file, "rb"))["bid-factor"]
    # else:
    #     bf = 1
    #
    # (auction, imp, clk, cost) = agent.run(auction_in, bid_log_path, N, c0,
    #                                           max_market_price, delimiter=" ", save_log=True, bid_factor=bf)
    #
    # win_rate = imp / auction * 100
    # cpm = (cost / 1000) / imp * 1000
    # ecpc = (cost / 1000) / clk
    # obj = opt_obj.get_obj(imp, clk, cost)
    # log = "{:<80}\t{:>10}\t{:>8}\t{:>10}\t{:>8}\t{:>8}\t{:>8.2f}%\t{:>8.2f}\t{:>8.2f}" \
    #     .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
    # print(log)
    # log_in.write(log + "\n")



