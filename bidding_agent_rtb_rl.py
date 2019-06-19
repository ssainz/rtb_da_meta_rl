# https://github.com/han-cai/rlb-dp/blob/master/python/bid_ls.py

# https://github.com/han-cai/rlb-dp/blob/master/python/lin_bid.py
from bidding_agent import bidding_agent
import pickle
import os
from utils import getTime
import sys

class bidding_agent_rtb_rl(bidding_agent):

    def init(self,  environment, camp_info, gamma):
        self.gamma = gamma
        self.init(environment, camp_info)




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