# https://github.com/han-cai/rlb-dp/blob/master/python/bid_ls.py

# https://github.com/han-cai/rlb-dp/blob/master/python/lin_bid.py

from bidding_agent import bidding_agent
import pickle
import os
from utils import getTime
import sys


class bidding_agent_rtb_rl_dp_tabular(bidding_agent):
    up_precision = 1e-10
    zero_precision = 1e-12

    def init(self, environment, camp_info, opt_obj, gamma):
        self.opt_obj = opt_obj
        self.gamma = gamma
        self.v1 = self.opt_obj.v1
        self.v0 = self.opt_obj.v0
        self.V = []
        self.D = []
        super(bidding_agent_rtb_rl_dp_tabular, self).init(environment, camp_info)

    def calc_optimal_value_function_with_approximation_i(self, N, B, max_bid, m_pdf, save_path):
        # print(getTime() + "\tvalue function with approx_i, N={}, B={}, save in {}".format(N, B, save_path))
        V_out = open(save_path, "w")
        V = [0] * (B + 1)
        nV = [0] * (B + 1)
        V_max = 0
        V_inc = 0
        if self.v0 != 0:
            a_max = min(int(self.v1 * self.theta_avg / self.v0), max_bid)
        else:
            a_max = max_bid
        for b in range(0, a_max + 1):
            V_inc += m_pdf[b] * (self.v1 * self.theta_avg - self.v0 * b)
        for n in range(1, N):
            a = [0] * (B + 1)
            bb = B - 1
            for b in range(B, 0, -1):
                while bb >= 0 and self.gamma * (V[bb] - V[b]) + self.v1 * self.theta_avg - self.v0 * (b - bb) >= 0:
                    bb -= 1
                if bb < 0:
                    a[b] = min(max_bid, b)
                else:
                    a[b] = min(max_bid, b - bb - 1)

            for b in range(0, B):
                V_out.write("{}\t".format(V[b]))
            V_out.write("{}\n".format(V[B]))

            V_max = self.gamma * V_max + V_inc
            flag = False
            for b in range(1, B + 1):
                nV[b] = self.gamma * V[b]
                for delta in range(0, a[b] + 1):
                    nV[b] += m_pdf[delta] * (
                        self.v1 * self.theta_avg + self.gamma * (V[b - delta] - V[b]) - self.v0 * delta)
                if abs(nV[b] - V_max) < self.up_precision:
                    for bb in range(b + 1, B + 1):
                        nV[bb] = V_max
                    flag = True
                    break
            V = nV[:]
            # if flag:
            #     print(getTime() + "\tround {} end with early stop.".format(n))
            # else:
            #     print(getTime() + "\tround {} end.".format(n))
        for b in range(0, B):
            V_out.write("{0}\t".format(V[b]))
        V_out.write("{0}\n".format(V[B]))
        V_out.flush()
        V_out.close()

    def calc_Dnb(self, N, B, max_bid, m_pdf, save_path):
        print(getTime() + "\tD(n, b), N={}, B={}, save in {}".format(N, B, save_path))
        D_out = open(save_path, "w")
        V = [0] * (B + 1)
        nV = [0] * (B + 1)
        V_max = 0
        V_inc = 0
        if self.v0 != 0:
            a_max = min(int(self.v1 * self.theta_avg / self.v0), max_bid)
        else:
            a_max = max_bid
        for b in range(0, a_max + 1):
            V_inc += m_pdf[b] * (self.v1 * self.theta_avg - self.v0 * b)
        for n in range(1, N):
            a = [0] * (B + 1)
            bb = B - 1
            for b in range(B, 0, -1):
                while bb >= 0 and self.gamma * (V[bb] - V[b]) + self.v1 * self.theta_avg - self.v0 * (b - bb) >= 0:
                    bb -= 1
                if bb < 0:
                    a[b] = min(max_bid, b)
                else:
                    a[b] = min(max_bid, b - bb - 1)

            for b in range(0, B):
                dtb = V[b + 1] - V[b]
                if abs(dtb) < self.zero_precision:
                    dtb = 0
                if b == B - 1:
                    D_out.write("{}\n".format(dtb))
                else:
                    D_out.write("{}\t".format(dtb))

            V_max = self.gamma * V_max + V_inc
            flag = False
            for b in range(1, B + 1):
                nV[b] = self.gamma * V[b]
                for delta in range(0, a[b] + 1):
                    nV[b] += m_pdf[delta] * (
                        self.v1 * self.theta_avg + self.gamma * (V[b - delta] - V[b]) - self.v0 * delta)
                if abs(nV[b] - V_max) < self.up_precision:
                    for bb in range(b + 1, B + 1):
                        nV[bb] = V_max
                    flag = True
                    break
            V = nV[:]
            if flag:
                print(getTime() + "\tround {} end with early stop.".format(n))
            else:
                print(getTime() + "\tround {} end.".format(n))
        for b in range(0, B):
            dtb = V[b + 1] - V[b]
            if abs(dtb) < self.zero_precision:
                dtb = 0
            if b == B - 1:
                D_out.write("{}\n".format(dtb))
            else:
                D_out.write("{}\t".format(dtb))
        D_out.flush()
        D_out.close()

    def load_value_function(self, N, B, model_path):
        self.V = [[0 for i in range(B + 1)] for j in range(N)]
        with open(model_path, "r") as fin:
            n = 0
            for line in fin:
                line = line[:len(line) - 1].split("\t")
                for b in range(B + 1):
                    self.V[n][b] = float(line[b])
                n += 1
                if n >= N:
                    break

    def load_Dnb(self, N, B, model_path):
        self.D = [[0 for i in range(B)] for j in range(N)]
        with open(model_path, "r") as fin:
            n = 0
            for line in fin:
                line = line[:len(line) - 1].split("\t")
                for b in range(B):
                    self.D[n][b] = float(line[b])
                n += 1
                if n >= N:
                    break

    def bid(self, n, b, theta, max_bid):
        a = 0
        if len(self.V) > 0:
            for delta in range(1, min(b, max_bid) + 1):
                #print("n = {}, b = {} ".format(n, b))
                if self.v1 * theta + self.gamma * (self.V[n - 1][b - delta] - self.V[n - 1][b]) - self.v0 * delta >= 0:
                    a = delta
                else:
                    break
        elif len(self.D) > 0:
            value = self.v1 * theta
            for delta in range(1, min(b, max_bid) + 1):
                value -= self.gamma * self.D[n - 1][b - delta] + self.v0
                if value >= 0:
                    a = delta
                else:
                    break

        return a

    def run(self, bid_log_path, N, c0, max_bid, input_type="file reader", delimiter=" ", save_log=False):
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

            action = self.bid(n, b, theta, max_bid)
            action = min(int(action), min(b, max_bid))

            done, new_theta, new_price, result_imp, result_click = self.env.step(action)

            log = getTime() + "\t{}\t{}_{}\t{}_{}_{}\t{}_{}\t".format(
                episode, b, n, action, price, result_click, clk, imp)
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

    def run_and_output_datasets(self, bid_log_path, N, c0, max_bid, input_type="file reader", delimiter=" ", save_log=False):
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

        X = []
        Y = []

        while not done:


            action = self.bid(n, b, theta, max_bid)
            action = min(int(action), min(b, max_bid))

            X.append([theta,price])
            Y.append([action])

            done, new_theta, new_price, result_imp, result_click = self.env.step(action)

            log = getTime() + "\t{}\t{}_{}\t{}_{}_{}\t{}_{}\t".format(
                episode, b, n, action, price, result_click, clk, imp)
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

            if n == 0:
                episode += 1
                n = N
                b = B

            theta = new_theta
            price = new_price

        if save_log:
            log_in.flush()
            log_in.close()

        return X, Y



