# https://github.com/han-cai/rlb-dp/blob/master/python/lin_bid.py
from bidding_agent import bidding_agent
import pickle
import os
from utils import getTime
import sys

class bidding_agent_linear(bidding_agent):

    def parameter_tune(self, opt_obj, save_path, N, c0, max_b0, max_bid, load=True):

        if load and os.path.isfile(save_path):
            var_map = pickle.load(open(save_path, "rb"))
            self.b0 = var_map["b0"]
            obj = var_map["best_obj"]

        else:
            obj = 0
            bb = 0
            tune_list = []
            kp_dc = 0
            for bc in range(self.step, max_b0 + self.step, self.step):
                self.b0 = bc
                (auction, imp, clk, cost) = self.run(None, N, c0, max_bid, save_log=False)
                perf = opt_obj.get_obj(imp, clk, cost)
                tune_list.append((bc, perf))
                if perf >= obj:
                    obj = perf
                    bb = bc
                    kp_dc = 0
                else:
                    kp_dc += 1
                if kp_dc >= 5:
                    break
            if bb == max_b0:
                bc = max_b0 + self.step
                while True:
                    self.b0 = bc
                    (auction, imp, clk, cost) = self.run(None, N, c0, max_bid,  save_log=False)
                    perf = opt_obj.get_obj(imp, clk, cost)
                    tune_list.append((bc, perf))
                    if perf >= obj:
                        obj = perf
                        bb = bc
                        bc += self.step
                    else:
                        break
                for _i in range(5):
                    bc += self.step
                    self.b0 = bc
                    (auction, imp, clk, cost) = self.run(None, N, c0, max_bid, save_log=False)
                    perf = opt_obj.get_obj(imp, clk, cost)
                    tune_list.append((bc, perf))
            self.b0 = bb
            pickle.dump({"b0": self.b0, "best_obj": obj, "tune_list": tune_list}, open(save_path, "wb"))
            print("Lin-Bid parameter tune: b0={}, best_obj={} and save in {}".format(self.b0, obj, save_path))

    def run(self, bid_log_path, N, c0, max_bid, save_log=False):

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
        theta, price = self.env.reset()
        done = False
        while not done:

            a = min(int(theta * self.b0 / self.theta_avg), max_bid)
            action = min(b, a)

            done, new_theta, new_price, result_imp, result_click = self.env.step(action)

            log = getTime() + "\t{}\t{}_{}\t{}_{}_{}\t{}\t".format(episode, b, n, a, result_click, clk, imp)

            if save_log:
                log_in.write(log + "\n")

            if result_imp == 1:
                imp += 1
                if result_click == 1:
                    clk += 1
                b -= price
                cost += price
            n -= 1
            auction += 1

            theta = new_theta
            price = new_price

            if n == 0:
                episode += 1
                n = N
                b = B

        if save_log:
            log_in.flush()
            log_in.close()

        return auction, imp, clk, cost