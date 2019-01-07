import argparse
import datetime
import json
import frozendict
import os

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
# import pandas as pd
from scipy.interpolate import interp1d
from types import SimpleNamespace

from pprint import pprint
from collections import defaultdict

from agent import RandomAgent, QLAgent, SQLAgent, MIRLAgent, SQL_mAgent
from environment import GridWorld

np.seterr(all='raise')


PAPER_COLORS = {
    'QLAgent': 'g',
    'SQLAgent': 'r',
    'SQL_mAgent': 'orange',
    'MIRLAgent': 'royalblue'
}


def eval_snapshot(agent_class, agent_checkpoint, world_conf, config):
    '''
    described in the paper: "The evaluation for a single snapshot is conducted by running the policy for 
    30 episodes lasting at most 100 environment steps. The epsilon value when in evaluation mode is set to 0.05.   
    Every individual experiment is repeated with 10 different initial random seeds and results are averaged"
    '''
    n_seeds = 10
    n_episodes = 30
    max_t = 100
    rewards = list()
    
    env = GridWorld(world_conf)
    agent = agent_class(env.get_actions(), config, **agent_checkpoint, mode='eval')
    
    for i_seed in range(n_seeds):
        np.random.seed()
        seed_results = list()
        
        for i_episode in range(n_episodes):
            episode_reward = 0
            obs = env.reset()
            
            for t in range(max_t):
                action = agent.choose_action(obs)
                reward, next_obs, done = env.action(action)
                
                obs = next_obs
                episode_reward += reward
    
                if done:
                    break
            
            seed_results.append(episode_reward)
            
        rewards.append(np.mean(seed_results))
      
    return np.mean(rewards)


def train(envs, agents, config):
    '''
    main training loop
    '''
    log_step = 1000

    # evaluation_data will store info for plot 1
    evaluation_data = defaultdict(dict)
    training_data = defaultdict(dict)

    # storing info for plot 2
    correct_info = defaultdict(dict)

    for envname, envconf in envs.items():
        env = GridWorld(envconf)
            
        for agent_class in agents:
            log_subfolder = os.path.join(log_folder, envname.replace(' ', '_'), agent_class.__name__)
            os.makedirs(log_subfolder)
            
            with open(os.path.join(log_subfolder, 'logs.csv'), 'w+') as f_log:
                print('\nTraining agent %s' % agent_class.__name__)
                agent = agent_class(env.get_actions(), config)

                total_reward = 0
                obs = env.reset()
                evals = dict()
                train = dict()
                correct_action = list()

                for global_step in range(config.max_steps):
                    action = agent.choose_action(obs)
                    reward, next_obs, done = env.action(action)

                    agent.update(obs, action, reward, next_obs, done)

                    if (global_step + 1) % 100 == 0:
                        eval_reward = eval_snapshot(agent_class, agent.get_checkpoint(), envconf, config)
                        evals[global_step] = eval_reward
                        print('Step %s, reward eval %s' % (global_step + 1, eval_reward))

                    if (global_step + 1) % log_step == 0:
                        agent.log(log_subfolder, global_step)
                        log_vars = ','.join([str(global_step), str(eval_reward)] + agent.get_current_vars()) + '\n'
                        f_log.write(log_vars)
                        
                    # correct action information (different evaluation for both gridworlds)
                    if envname == 'Grid_World_3x20':        # log correct action at each step
                        correct_action.append(1 if action == 1 else 0)
                    elif envname == 'Grid_World_8x8':       # log correct action at state (3,6)
                        x_agent, y_agent = np.where(obs == 3)
                        if (x_agent[0], y_agent[0]) == (3, 6):
                            correct_action.append(1 if action == 3 else 0)
                        
                    total_reward += reward
                    obs = next_obs
                    # env.display()
                    
                    if done:
                        # print("Env done with reward %s" % total_reward)
                        train[global_step] = total_reward
                        total_reward = 0
                        obs = env.reset()

            evaluation_data[envname][agent_class.__name__] = evals
            training_data[envname][agent_class.__name__] = train
            correct_info[envname][agent_class.__name__] = correct_action
                
            with open(os.path.join(log_subfolder, 'eval_data.json'), 'w+') as f_dump:
                f_dump.write(json.dumps(evals))
            
            with open(os.path.join(log_subfolder, 'correct_action.json'), 'w+') as f_correct:
                f_correct.write(json.dumps({'data': correct_action}))

    return evaluation_data, correct_info


def plot_results(log_folder, evaluation_data):
    '''
    reward over time
    '''
    for envname, env_eval_data in evaluation_data.items():
        for agent_type, eval_data in env_eval_data.items():
            data_x = np.array(list(eval_data.keys()))
            data_y = np.array(list(eval_data.values()))
            x_smooth = np.linspace(data_x.min(), data_x.max(), 200)
            f_smooth = interp1d(data_x, data_y, kind='cubic')
            df = pd.Series(data_y)
            plt.plot(data_x, df.ewm(span=10).mean(), PAPER_COLORS[agent_type])

            plt.plot(x_smooth, f_smooth(x_smooth), PAPER_COLORS[agent_type], label='%s eval RAW' % agent_type, alpha=0.4)
            # plt.plot(ewma(data_x, span=1000), label='%s train EWMA' % agent_type)

        plt.xlabel('environment interactions')
        plt.ylabel('reward')
        plt.legend()
        plt.title(envname)

        plt.savefig(log_folder + '/results.png')


def plot_correct_action(log_folder, correct_info):
    '''
    figure 1 last column
    differs for the two environments
    '''

    labels = {
        'Grid_World_3x20': {
            'title': 'Correct action while training',
            'x_label': 'Training step',
            'y_label': '1: Correct action (right), 0: Incorrect action'
        },
        'Grid_World_8x8': {
            'title': 'Correct Infrequent action',
            'x_label': 'Interactions in state (3, 6)',
            'y_label': '1: Correct action (left), 0: Incorrect action'
        } 
    }

    for envname, correct_data in correct_info.items():
        for agent_type, correct_action in correct_data.items():
            data_x = np.arange(0, len(correct_action), 1)
            data_y = np.array(correct_action)
            df = pd.Series(correct_action)
            plt.plot(data_x, df.ewm(span=100).mean(), PAPER_COLORS[agent_type], label=agent_type)

        plt.xlabel(labels[envname]['x_label'])
        plt.ylabel(labels[envname]['y_label'])
        plt.legend()
        plt.title(labels[envname]['title'])

        plt.savefig(log_folder + '/correct_action.png')


def plot(log_folder, evaluation_data, correct_info):
    '''
    plotting reward over time and correct action
    '''
    plot_results(log_folder, evaluation_data)

    plot_correct_action(log_folder, correct_info)


# python3 main.py --root_dir /home/caleml/grotile --exp_name 100k_mirl --max_steps 100000 --agents MIRLAgent
# python3 main.py --root_dir /home/caleml/grotile --exp_name 100k_mirl_3x20 --max_steps 100000 --agents MIRLAgent --envs Grid_World_3x20
if __name__ == '__main__':

    parser = argparse.ArgumentParser("tabular")

    parser.add_argument("--root_dir", required=True, type=str, help="base directory")
    parser.add_argument("--exp_name", required=True, type=str, help="help sorting experiment folders")
    parser.add_argument("--max_steps", required=True, type=int, help="total number of interactions with the environment")
    parser.add_argument("--agents", type=str, default='QLAgent,SQLAgent,SQL_mAgent,MIRLAgent')
    parser.add_argument("--envs", type=str, default='Grid_World_3x20,Grid_World_8x8')

    args = parser.parse_args()

    world1 = {
        'size': (3, 20),
        'reward': (1, 18),
        'walls': []
    }

    world2 = {
        'size': (8, 8),
        'reward': (3, 5),
        'walls': [(2, 4), (2, 5), (3, 4), (4, 4), (4, 5)]
    }

    config = frozendict.frozendict({
        'eps_train': 0.1,
        'eps_eval': 0.05,
        'gamma': 0.999,
        'rho_lr': 1e-3,
        'beta_lr': 2e-3,   # use?
        'c': 1e-3,
        'omega': 0.8,
        'max_steps': args.max_steps
    })

    config = SimpleNamespace(**config)

    base_envs = {
        'Grid_World_3x20': world1,
        'Grid_World_8x8': world2
    }

    # TODO make that a better argparse
    user_agents = args.agents.split(",")
    agents = [c for c in [QLAgent, SQLAgent, SQL_mAgent, MIRLAgent] if c.__name__ in user_agents]
    user_envs = args.envs.split(',')
    envs = {k: v for k, v in base_envs.items() if k in user_envs}


    log_folder = args.root_dir + '/tabular/logs/%s_%s' % (datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), args.exp_name)
    print('Creating experiments in %s' % log_folder)

    evaluation_data, correct_info = train(envs, agents, config)
    plot(log_folder, evaluation_data, correct_info)


