#!/usr/bin/env python
# -*- coding: utf-8 -*-


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.p = self.initial_p

    @property
    def value(self):
        return self.p

    def reset(self):
        self.p = self.initial_p

    def update(self, t):
        """See Schedule.value"""
        if self.p > self.final_p:
            fraction = min(float(t) / self.schedule_timesteps, 1.0)
            self.p = self.initial_p + fraction * (self.final_p - self.initial_p)
