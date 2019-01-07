# -*- coding: utf-8 -*-

import logging

log = logging.getLogger("main")


class Trainer(object):
    """
    Base class for the trainer of the agent
    """

    def __init__(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


def train(agent, replay, env, logger, config):
    log.info("Begin training with %s on %s" % (agent.scope, env.spec.id))
    agent.update()
    s = env.reset()

    ep = 0
    rw = 0
    last_rw = 0
    loss = 0.

    for t in range(config.max_steps):
        a, q = agent.act(s)
        s1, r, d, _ = env.step(a)
        replay.add(s, a, r, s1, d)
        rw += r
        s = s1
        if d:
            s = env.reset()
            ep += 1
            last_rw = rw
            rw = 0
        if t % config.summarize_every == 0 and len(replay)>0:
            logger.add({'t': t, 'ep': ep, 'rw': last_rw, 'q': q, 'loss': loss, 'eps': agent.eps, 'beta': agent.beta})
            # logger.log()
            # stats = logger.summarize()

        if len(replay) > config.warmup and t % config.train_every == 0:
            batch = replay.sample(batch_size=config.batch_size)
            loss = agent.train(batch)

        if len(replay) > config.warmup and t % config.update_every == 0:
            agent.update()
            agent.summarize(batch)  # this push everything in the tensoboard

        if t > 0 and t % config.save_every == 0:
            agent.save()
            logger.dump()

    agent.save()
    replay.dump()
    logger.dump()

    log.info("Training finished, Summary saved at: %s" % logger.main_path)


def test(agent, env, logger, config):
    log.info("Begin Test with %s on %s" % (agent.scope, env.spec.id))
    env.seed(config.test_seed)
    rw = 0
    ep = 0
    s = env.reset()
    logger.reset(mode='test', stats=('t', 'ep', 'rw', 'q'))
    env.render()
    for t in range(config.test_steps):
        a, q = agent.act(s)
        s1, r, d, _ = env.step(a)
        env.render()
        rw += r
        s = s1
        if d:
            s = env.reset()
            ep += 1
            logger.add({'t': t, 'ep': ep, 'rw': rw, 'q': q})
            logger.log()
            rw = 0
    logger.log()
    logger.dump()
    log.info("Test finished, Summary saved at: %s" % logger.main_path)
