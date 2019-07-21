import bidding_agent
import os
import json
import argparse
import torch
import numpy as np
from utils import getTime

from maml_rl.metalearner import MetaLearner
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
from maml_rl.policies.normal_mlp import NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler

from tensorboardX import SummaryWriter

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

class bidding_agent_meta(bidding_agent):

    def __init__(self):
        print("bidding_agent_meta created!")

    def run_meta_training(self, final_model_folder):

        parser = argparse.ArgumentParser(description='Reinforcement learning with '
                                                     'Model-Agnostic Meta-Learning (MAML)')

        # General
        parser.add_argument('--env-name', type=str,
                            help='name of the environment', default='bidding')
        parser.add_argument('--gamma', type=float, default=0.95,
                            help='value of the discount factor gamma')
        parser.add_argument('--tau', type=float, default=1.0,
                            help='value of the discount factor for GAE')
        parser.add_argument('--first-order', action='store_true',
                            help='use the first-order approximation of MAML')

        # Policy network (relu activation function)
        parser.add_argument('--hidden-size', type=int, default=100,
                            help='number of hidden units per layer')
        parser.add_argument('--num-layers', type=int, default=2,
                            help='number of hidden layers')

        # Task-specific
        parser.add_argument('--fast-batch-size', type=int, default=20,
                            help='batch size for each individual task')
        parser.add_argument('--fast-lr', type=float, default=0.5,
                            help='learning rate for the 1-step gradient update of MAML')

        # Optimization
        parser.add_argument('--num-batches', type=int, default=10,
                            help='number of batches')
        parser.add_argument('--meta-batch-size', type=int, default=10,
                            help='number of tasks per batch')
        parser.add_argument('--max-kl', type=float, default=1e-2,
                            help='maximum value for the KL constraint in TRPO')
        parser.add_argument('--cg-iters', type=int, default=10,
                            help='number of iterations of conjugate gradient')
        parser.add_argument('--cg-damping', type=float, default=1e-5,
                            help='damping in conjugate gradient')
        parser.add_argument('--ls-max-steps', type=int, default=15,
                            help='maximum number of iterations for line search')
        parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
                            help='maximum number of iterations for line search')

        # Miscellaneous
        parser.add_argument('--output-folder', type=str, default='maml',
                            help='name of the output folder')
        parser.add_argument('--num-workers', type=int, default=5,
                            help='number of workers for trajectories sampling')
        parser.add_argument('--device', type=str, default='cpu',
                            help='set the device (cpu or cuda)')

        args = parser.parse_args()

        continuous_actions = (args.env_name in ['bidding', 'AntVel-v1', 'AntDir-v1',
                                                'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
                                                '2DNavigation-v0'])

        # Create logs and saves folder if they don't exist
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        if not os.path.exists('./saves'):
            os.makedirs('./saves')
        # Device
        args.device = torch.device(args.device
                                   if torch.cuda.is_available() else 'cpu')

        writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
        save_folder = './saves/{0}'.format(args.output_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with open(os.path.join(save_folder, 'config.json'), 'w') as f:
            config = {k: v for (k, v) in vars(args).items() if k != 'device'}
            config.update(device=args.device.type)
            json.dump(config, f, indent=2)

        sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
                               num_workers=args.num_workers)
        if continuous_actions:
            policy = NormalMLPPolicy(
                int(np.prod(sampler.envs.observation_space.shape)),
                int(np.prod(sampler.envs.action_space.shape)),
                hidden_sizes=(args.hidden_size,) * args.num_layers)
            self.policy = policy
        else:
            policy = CategoricalMLPPolicy(
                int(np.prod(sampler.envs.observation_space.shape)),
                sampler.envs.action_space.n,
                hidden_sizes=(args.hidden_size,) * args.num_layers)
        baseline = LinearFeatureBaseline(
            int(np.prod(sampler.envs.observation_space.shape)))

        metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
                                  fast_lr=args.fast_lr, tau=args.tau, device=args.device)

        for batch in range(args.num_batches):
            tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
            episodes = metalearner.sample(tasks, first_order=args.first_order)
            metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
                             cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
                             ls_backtrack_ratio=args.ls_backtrack_ratio)

            # Tensorboard
            writer.add_scalar('total_rewards/before_update',
                              total_rewards([ep.rewards for ep, _ in episodes]), batch)
            writer.add_scalar('total_rewards/after_update',
                              total_rewards([ep.rewards for _, ep in episodes]), batch)

            # Save policy network
            final_model_path = final_model_folder + "meta_rl_gamma_policy_{}.pt".format(batch)
            with open(final_model_path, 'wb') as f:
                torch.save(policy.state_dict(), f)
            return final_model_path

    def load_model(self, final_model_path):

        self.policy.load_state_dict(torch.load(final_model_path))

    def run(self, env, bid_log_path, N, c0, max_bid, save_log=False):

        self.env = env

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

            observation = np.zeros(2, dtype=np.float32)
            observation[0] = theta
            observation[1] = price

            observations_tensor = torch.from_numpy(observation)
            with torch.no_grad():
                actions_tensor = self.policy(observations_tensor).sample()
                a = actions_tensor.cpu().numpy()

            action = a
            #action = min(b, a)
            #a = min(int(theta * self.b0 / self.theta_avg), max_bid)


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