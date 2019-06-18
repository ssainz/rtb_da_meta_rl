import config
import sys
from utils import Opt_Obj
from bidding_environment import BidEnv
from bidding_agent_linear import bidding_agent_linear


obj_type = "clk"
clk_vp = 1
N = 1000
c0 = 1 / 8
gamma = 1

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
    auction_in = open(data_path + camp + "/test.theta.txt", "r")
    opt_obj = Opt_Obj(obj_type, int(clk_vp * camp_info["cost_train"] / camp_info["clk_train"]))
    B = int(camp_info["cost_train"] / camp_info["imp_train"] * c0 * N)

    # Lin-Bid

    env = BidEnv(camp_info, auction_in)

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

