# -*- coding: utf-8 -*-

import os
from types import SimpleNamespace
import logging
import argparse

import gym
from gym.wrappers import Monitor
import gym_minigrid

from gym_minigrid.wrappers import ImgObsWrapper
from utils.gym_utils import make_atari
from config import *
from memory import ReplayBuffer
from logger import Logger
from train import train, test
from agent import DQN, SQL, MIRL, GL

logging.getLogger("main")


def which_agent(name):
    if name == "sql":
        return SQL
    elif name == "dqn":
        return DQN
    elif name == "mirl":
        return MIRL
    elif name == "gl":
        return GL
    else:
        raise NotImplementedError("Nope.")


def which_env(name):
    if 'Grid' in name:
        env = ImgObsWrapper(gym.make(name))
        test_env = ImgObsWrapper(gym.make(name))
    else:
        env = make_atari(name)
        test_env = make_atari(name)
    return env, test_env, (env.observation_space.shape, env.action_space.n)


def main(agent, env, gpu, gpu_ratio, seed=0):
    gridworld_config.update({'agent': agent, 'env': env})
    config = SimpleNamespace(**gridworld_config)
    config.seed = int(seed)

    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # TODO this should all go in a proper training class
    logger = Logger(base_path=config.path, config=gridworld_config)
    memory = ReplayBuffer(size=config.replay_size, path=logger.main_path, seed=config.seed)

    # TODO meh
    config.restore_path = logger.main_path

    env, test_env, env_specs = which_env(env)
    agent = which_agent(agent)(*env_specs, config)

    agent.build(gpu_ratio)
    logging.debug("Model built")
    logger.register(("t", "ep", "rw", "q", "eps", "beta", "loss"))

    if config.mode == "train":
        try:
            env.seed(config.seed)
            train(agent, memory, env, logger, config)
        except KeyboardInterrupt:
            agent.save()
            logging.info("Training interrupted")
    test_env.seed(config.test_seed)
    test_env = Monitor(test_env, directory=logger.paths['test']) if config.monitor else test_env
    test(agent, test_env, logger, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("RL")

    parser.add_argument("--agent", default="mirl", type=str, help="sql / dqn / mirl / gl")
    parser.add_argument("--env", default="MiniGrid-Empty-6x6-v0", type=str, help="Check utils/env.txt")
    parser.add_argument("--gpu", default=0, type=int,
                        help="# of the GPU on which to run the experiment. For running on machines with more than one GPU, be sure to add the gpus in the docker-compose.yml")
    parser.add_argument("--gpu_ratio", type=int, help="percentage of the GPU memory to allocate to the experiment")
    parser.add_argument("--seed", default=0, type=int, help="training seed")

    print(str(vars(parser.parse_args())))

    main(**vars(parser.parse_args()))
