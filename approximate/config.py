# -*- coding: utf-8 -*-

gridworld_config = {
    'seed': 42,
    'test_seed': 11332,
    'mode': "train",
    'path': "logs",
    'monitor': True,
    'batch_size': 32,  # 128,
    'n_train_repeat': 1,
    'train_every': 4,
    'summarize_every': int(1e3),
    'save_every': int(1e4),
    'replay_size': int(1e6),
    'max_steps': int(1e6),
    'test_steps': int(1e3),
    'update_every': int(1e4),
    'warmup': int(1e3),
    'gamma': 0.99,
    'q_lr': 2e-5,
    'beta_lr': 5e-3,# assuming linear schedule
    'beta': 0.01,
    'eps': 0.01,
    'rho_lr': 2e-6,
    'grad_norm': 10.,
    'expl_fraction': 0.2,  # define the amount of the timestep
    'model':  # as in deepmind paper?
        {'conv': [(32, 8, 4), (64, 4, 2), (64, 3, 1)], 'fc': 256},
    'stats': ('t', 'ep', 'rw', 'q', 'loss', 'eps', 'beta')

}

atari_config = {
    'seed': 3,
    'path': 'logs/',
    'batch_size': 32,  # 128,
    'n_train_repeat': 1,
    'train_every': 4,
    'summarize_every': 100,  # ep
    'save_every': 100,
    'replay_size': int(1e6),
    'max_steps': int(1e7),
    'update_every': int(1e4),
    'warmup': int(1e4),
    'gamma': 0.99,
    'q_lr': 2e-5,
    'beta_lr': 3.3e-6,
    'beta': 0.01,
    'beta_bound': 2e2,
    'v_bound': 1e4,
    'eps': 0.01,
    'rho_lr': 2e-6,
    'grad_norm': 10.,
    'expl_fraction': 0.1,  # define the amount of the timestep
    'model':  # as in deepmind paper?
        {'conv': [(32, 8, 4), (64, 4, 2), (64, 3, 1)], 'fc': 256},

}
