import collections
import copy
import json
import os

import numpy as np

np.seterr(all='raise')


class TabularAgent:
    '''
    Base class for tabular agents
    '''

    def __init__(self, possible_actions, config, q_table=None, mode='train'):
        self.possible_actions = possible_actions
        self.omega = config.omega
        self.gamma = config.gamma

        if mode == 'train':
            self.epsilon = config.eps_train
        elif mode == 'eval':
            self.epsilon = config.eps_eval
        else:
            raise Exception("Unknown mode %s" % mode)

        self.q_table = q_table if q_table else collections.defaultdict(lambda: 0)  # authors init q table to 0 
        self.n_table = collections.defaultdict(lambda: 0)

    def get_key(self, obs, a):
        '''
        get str q table key for (obs, a) pair
        '''
        return "%s_%s" % (str(obs), str(a)) 

    def get_checkpoint(self):
        return {
            'q_table': copy.deepcopy(self.q_table)
        }

    def log(self, log_folder, global_step):
        '''
        log the q table so it can be retrieved later
        '''
        log_path = os.path.join(log_folder, "qtable_%s.json" % global_step)
        with open(log_path, "w+") as f_qlog:
            f_qlog.write(json.dumps(self.q_table))

    def get_current_vars(self):
        '''
        for logging
        '''
        return []

    def get_exp(self, obs, a):
        '''
        put here but only for SQL, SQL_m and MIRL since QL doesn't use soft exp...
        the inner value is bounded by [-700, 700] to avoid under/overflow
        '''
        raw_val = self.beta * self.q_table[self.get_key(obs, a)]
        return np.exp(min(max(-300.0, raw_val), 300))

    def choose_action(self, current_obs):
        raise NotImplementedError

    def update(self, current_obs, action, reward, next_obs, done):
        raise NotImplementedError


class RandomAgent(TabularAgent):
    '''
    for testing
    '''
    def choose_action(self, current_obs):
        return np.random.choice(self.possible_actions)

    def update(self, current_obs, action, reward, next_obs, done):
        pass


class QLAgent(TabularAgent):
    '''
    traditional Q learning with epsilon greedy behaviour policy
    '''

    def choose_action(self, current_obs):
        '''
        epsilon-greedy
        '''
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.possible_actions)
        else:
            q_values = np.array([self.q_table[self.get_key(current_obs, a)] for a in self.possible_actions])
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))

    def get_exp(self, obs, a):
        raise NotImplementedError

    def update(self, current_obs, action, reward, next_obs, done):
        '''
        adaptive learning rate for q values: alpha_q = n(s, a)^{-omega}
        q_new = q_old * alpha_q [r + gamma * max_a Q(S_t+1, a) - q_old]
        '''
        s_a = self.get_key(current_obs, action)
        self.n_table[s_a] += 1

        alpha_q = self.n_table[s_a] ** (-self.omega)

        old_q = self.q_table[s_a]
        next_q = [self.q_table[self.get_key(next_obs, a)] for a in self.possible_actions]
        new_q = old_q + alpha_q * (reward + self.gamma * max(next_q) - old_q)

        self.q_table[s_a] = new_q


class SQLAgent(TabularAgent):
    '''
    Soft Q learning agent
    Q update is the same as equation 12 from MIRL except that the prior here is uniform
    '''

    def __init__(self, possible_actions, config, q_table=None, mode='train'):
        self.rho = np.ones(len(possible_actions)) / len(possible_actions)
        self.beta = config.c    # just so it isn't init to 0
        self.c = config.c       # constant for updating beta

        TabularAgent.__init__(self, possible_actions, config, q_table=q_table, mode=mode)

    def get_current_vars(self):
        return [str(self.beta)]

    def choose_action(self, current_obs):
        '''
        epsilon-greedy (not sure they used epsilon greedy for SQL experiment, awaiting answer)
        Author precision: behaviour policy is the argmax of soft q-values
        Quote: "The behaviour policy is the argmax of soft q-values. This is equivalent to using the argmax of the Boltzmann poliy as in section 4.1 since the prior in SQL is uniform and exp strictly monotonically increasing."
        '''
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.possible_actions)
        else:
            soft_q_values = np.array([self.rho[a] * self.get_exp(current_obs, a) for a in self.possible_actions])
            return np.random.choice(np.flatnonzero(soft_q_values == soft_q_values.max()))

    def update(self, current_obs, action, reward, next_obs, done):
        '''
        same adaptive learning rate for q values: alpha_q = n(s, a)^{-omega}
        soft update: q_new = q_old * alpha_q [r + gamma * max_a Q(S_t+1, a) - q_old]
        '''
        s_a = self.get_key(current_obs, action)
        old_q = self.q_table[s_a]
        self.n_table[s_a] += 1

        alpha_q = self.n_table[s_a] ** (-self.omega)
        t_soft = reward + (self.gamma / self.beta) * np.log(np.sum([self.rho[a] * self.get_exp(next_obs, a) for a in self.possible_actions]))

        new_q = old_q + alpha_q * (t_soft - old_q)

        self.q_table[s_a] = new_q
        self.beta += self.c


class SQL_mAgent(SQLAgent):
    '''
    Soft Q learning agent with marginal distribution over action instead of uniform for exploration
    '''

    def __init__(self, possible_actions, config, q_table=None, action_counters=None, mode='train'):
        SQLAgent.__init__(self, possible_actions, config, q_table=q_table, mode=mode)

        self.action_counters = np.zeros(len(self.possible_actions)) if action_counters is None else action_counters
        self.random_counter = 0

    def get_checkpoint(self):
        return {
            'q_table': copy.deepcopy(self.q_table),
            'action_counters': copy.deepcopy(self.action_counters)
        }

    def get_marginal_dist(self):
        total = self.action_counters.sum()
        if total == 0:
            return np.ones(len(self.possible_actions)) / len(self.possible_actions)

        return self.action_counters / total

    def choose_action(self, current_obs):
        '''
        exploration: use the marginal distribution over actions
        '''
        if np.random.random() <= self.epsilon:
            action = np.random.choice(self.possible_actions, p=self.get_marginal_dist())
            self.random_counter += 1
        else:
            soft_q_values = np.array([self.rho[a] * self.get_exp(current_obs, a) for a in self.possible_actions])
            action = np.random.choice(np.flatnonzero(soft_q_values == soft_q_values.max()))

        self.action_counters[action] += 1
        return action


class MIRLAgent(TabularAgent):
    '''
    MIRL tabular agent
    Q update is equation 12
    rho update is equation 11
    '''

    def __init__(self, possible_actions, config, q_table=None, mode='train'):
        self.rho = np.ones(len(possible_actions)) / len(possible_actions)
        self.beta = config.c    # just so it isn't init to 0
        self.c = config.c       # constant for updating beta
        self.rho_lr = config.rho_lr

        TabularAgent.__init__(self, possible_actions, config, q_table=q_table, mode=mode)

    def get_current_vars(self):
        return [str(self.beta)]

    def get_pi(self, current_obs):
        '''
        get pi(a|s), s being current_obs
        follows eq 8
        '''
        inner = self.rho * np.array([self.get_exp(current_obs, a) for a in self.possible_actions])
        z = inner.sum()
        return inner / z

    def choose_action(self, current_obs):
        '''
        uses the MIRL "epsilon-greedy"-like policy ("Behavioural policy" section in 4.1)
        '''
        try:
            if np.random.random() <= self.epsilon:
                return np.random.choice(self.possible_actions, p=self.rho)
            else:
                pi = self.get_pi(current_obs)
                # return np.argmax(pi)   numpy argmax doesn't break ties arbitrarily
                return np.random.choice(np.flatnonzero(pi == pi.max()))
        except Exception as e:
            print(str(e))
            print(type(e))
            import pdb; pdb.set_trace()

    def update(self, current_obs, action, reward, next_obs, done):
        '''
        same adaptive learning rate for q values: alpha_q = n(s, a)^{-omega}
        soft update: q_new = q_old * alpha_q [r + gamma * max_a Q(S_t+1, a) - q_old]
        '''
        try:
            s_a = self.get_key(current_obs, action)
            old_q = self.q_table[s_a]
            self.n_table[s_a] += 1

            alpha_q = self.n_table[s_a] ** (-self.omega)
            t_soft = reward + (self.gamma / self.beta) * np.log(np.sum([self.rho[a] * self.get_exp(next_obs, a) for a in self.possible_actions]))

            # eq 12
            new_q = old_q + alpha_q * (t_soft - old_q)

            # eq 11
            pi = self.get_pi(current_obs)
            new_rho = np.multiply((1 - self.rho_lr), self.rho) + np.multiply(self.rho_lr, pi)
            assert pi.shape == (4,)
            assert new_rho.shape == (4,)

            self.q_table[s_a] = new_q
            self.rho = new_rho
            self.beta += self.c

        except Exception as e:
            print(str(e))
            print(type(e))
            import pdb; pdb.set_trace()