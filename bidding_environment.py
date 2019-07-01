import numpy as np
import random as random
import config

class BidEnv():

    def __init__(self, info, auction_in_file):
        auction_in = open(auction_in_file, 'r')

        self._step = 0

        self.info = info
        self.auction_in = auction_in

        self.episodes = []

        for line in auction_in:
            line = line[:len(line) - 1].split(config.delimiter)
            click = int(line[0])
            price = int(line[1])
            theta = float(line[2])

            self.episodes.append((click, price, theta))


    def reset(self):
        self._step = 0

        (self.click, self.price, self.theta) = self.episodes[self._step]

        self._step += 1

        return self.theta, self.price

    def step(self, action):

        ret_click = 0
        ret_impression = 0
        if action >= self.price:
            ret_impression = 1
            ret_click = self.click


        (self.click, self.price, self.theta) = self.episodes[self._step]
        self._step += 1

        done = self._step >= len(self.episodes)

        return done, self.theta, self.price, ret_impression, ret_click

