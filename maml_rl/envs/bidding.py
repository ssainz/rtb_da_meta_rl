import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import config


class BiddingMDPEnv(gym.Env):
    """Bidding in display advertising.

    Follows a given simulation day and keeps state of which bid is next one and how much reward has happened.
    """

    def __init__(self):
        super(BiddingMDPEnv, self).__init__()
        #auction_in = open(auction_in_file, 'r')

        self._step = 0

        self.observation_space = spaces.Box(low=0, high=np.inf,
            shape=(2,), dtype=np.int32)
        self.action_space = spaces.Box(low=0.0, high=np.inf,
            shape=(1,), dtype=np.float32)

        #self.info = info
        #self.auction_in = auction_in

        self.episodes = []

        #for line in auction_in:
        #    line = line[:len(line) - 1].split(config.delimiter)
        #    click = int(line[0])
        #    price = int(line[1])
        #    theta = float(line[2])
        #    self.episodes.append((click, price, theta))


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):

        # build list of potential campaigns
        task_space = []
        for camp in config.ipinyou_camps:
            if camp != config.ipinyou_camps_target:
                task_space.append(camp)

        tasks = []
        for _ in range(num_tasks):
            random_camp = task_space[ np.random.randint(0,len(task_space)) ]
            file_location = config.ipinyouPath + random_camp + "/test.theta.txt"
            tasks.append({'camp': random_camp,
                          'file_location': file_location})

        return tasks

    def sample_target_task(self, N):

        # build list of potential campaigns
        task_space = []
        for camp in config.ipinyou_camps:
            if camp == config.ipinyou_camps_target:
                task_space.append(camp)

        tasks = []
        for _ in range(1):
            random_camp = task_space[ np.random.randint(0,len(task_space)) ]
            file_location = config.ipinyouPath + random_camp + "/test.theta.txt"
            tasks.append({'camp': random_camp,
                          'file_location': file_location,
                          'early_stop': N})

        return tasks

    def reset_task(self, task):
        self._task = task
        self._camp = task['camp']
        self._file_location = task['file_location']

        self._early_stop = None
        if 'early_stop' in task:
            self._early_stop = task['early_stop']
        self._file_contents = open(self._file_location, 'r')

        # Load the campaing.
        self.episodes = []

        for line in self._file_contents:
            line = line[:len(line) - 1].split(config.delimiter)
            click = int(line[0])
            price = int(line[1])
            theta = float(line[2])
            self.episodes.append((click, price, theta))
        self._file_contents.close()
        self._step = 0

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

        observation = np.zeros(2, dtype=np.float32)
        observation[0] = self.theta
        observation[1] = self.price

        reward = ret_impression + ret_click

        if self._early_stop is not None:
            if self._early_stop < self._step:
                done = True

        return observation, reward, done, self._task



