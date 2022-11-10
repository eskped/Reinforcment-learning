from pickle import NONE
import sys
import time
from constants import *
from environment import *
from state import State
import numpy as np
import math


"""
solution.py

This file is a template you should use to implement your solution.

You should implement code for each of the TODO sections below.

COMP3702 2022 Assignment 3 Support Code

Last updated by njc 12/10/22
"""

"""
I have used tutorl 9 as a reference for this assignment
"""


class RLAgent:

    #
    # TODO: (optional) Define any constants you require here.
    #

    def __init__(self, environment: Environment):
        self.environment = environment
        self.states = self.bfs()
        self.alpha = environment.alpha
        self.gamma = environment.gamma
        self.epsilon = 0.1
        self.N = 1
        self.n_a = [0, 0, 0, 0]
        self.n_sa = dict((state, [0, 0, 0, 0]) for state in self.states)
        # self.q_sa = dict((state, [None, None, None, None])
        #                  for state in self.states)
        self.q_table = dict((state, [0, 0, 0, 0]) for state in self.states)
        self.n_episodes = 50000
        self.max_timestep_per_episode = 10000
        self.q_rewards = []

        self.sarsa_q_table = dict((state, [0, 0, 0, 0])
                                  for state in self.states)
        self.sarsa_n_episodes = 15000
        self.sarsa_max_timestep_per_episode = 1000
        self.sarsa_rewards = []

    # === Q-learning ===================================================================================================

    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        # for ep in range(self.n_episodes):
        while self.environment.get_total_reward() > self.environment.training_reward_tgt*1.05:
            s = self.environment.get_init_state()  # initialize state
            episode_reward = 0
            for t in range(self.max_timestep_per_episode):
                a = self.epsilon_greedy(s)
                # _Environment__apply_dynamics
                r, s_next = self.environment.perform_action(
                    s, a)
                episode_reward += r
                old_q = self.q_table[s][a]
                best_next = self._get_best_action(s_next, self.q_table)
                best_next_q = self.q_table[s_next][best_next]
                if self.environment.is_solved(s_next):
                    best_next_q = 0
                target = r + self.gamma * best_next_q
                new_q = old_q + self.alpha * (target - old_q)
                self.q_table[s][a] = new_q
                # tabular Q-learning update
                # self.q_table[s][a] += self.alpha * \
                #     (r + self.gamma *
                #      np.max(self.q_table[s_next]) - self.q_table[s][a])
                s = s_next
                if self.environment.is_solved(s):
                    break

            length = len(self.q_rewards)
            if length >= 50:
                self.q_rewards.append(
                    (sum(self.q_rewards[-50:-1]) + episode_reward)/50)
            else:
                self.q_rewards.append((
                    sum(self.q_rewards[-length:-1]) + episode_reward)/(length+1))
        file = open("alpha0001.txt", "w+")
        content = str(self.q_rewards)
        file.write(content)
        file.close()

    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """

        return np.argmax(self.q_table[state])

    # === SARSA ========================================================================================================

    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """
        a = np.random.choice(ROBOT_ACTIONS)
        # for ep in range(self.sarsa_n_episodes):
        while self.environment.get_total_reward() > self.environment.training_reward_tgt*1.05:
            s = self.environment.get_init_state()  # initialize state
            a = self.epsilon_greedy(s)
            episode_reward = 0
            for t in range(self.sarsa_max_timestep_per_episode):
                r, s_next = self.environment.perform_action(s, a)
                episode_reward += r
                a2 = np.random.choice(ROBOT_ACTIONS)
                a2 = self._get_best_action(s_next, self.sarsa_q_table)
                self.sarsa_q_table[s][a] += self.alpha * \
                    (r + self.gamma *
                     self.sarsa_q_table[s_next][a2] - self.sarsa_q_table[s][a])
                s = s_next
                a = a2
                if self.environment.is_solved(s):
                    break
        #     length = len(self.sarsa_rewards)
        #     if length >= 50:
        #         self.sarsa_rewards.append(
        #             (sum(self.sarsa_rewards[-50:-1]) + episode_reward)/50)
        #     else:
        #         self.sarsa_rewards.append((
        #             sum(self.sarsa_rewards[-length:-1]) + episode_reward)/(length+1))
        #     episode_reward = 0

        # file = open("SARSA_rewards.txt", "w+")
        # content = str(self.sarsa_rewards)
        # file.write(content)
        # file.close()

    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """

        return np.argmax(self.sarsa_q_table[state])

    # === Helper Methods ===============================================================================================

    def bfs(self):
        states = []
        states.append(self.environment.get_init_state())
        frontier = [self.environment.get_init_state()]
        while len(frontier) > 0:
            current_state = frontier.pop()
            for action in ROBOT_ACTIONS:
                _, new_state = self.environment._Environment__apply_dynamics(
                    current_state, action)
                if new_state not in states and new_state not in frontier:
                    frontier.append(new_state)
            if current_state not in states:
                states.append(current_state)
        return states

    def epsilon_greedy(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.choice(ROBOT_ACTIONS)
        else:
            return np.argmax(self.q_table[s])

    def UCB1(self, s):
        unvisited = set()
        C = 2
        best_u = float('-inf')
        best_a = None
        for a in ROBOT_ACTIONS:
            if self.n_sa[s][a] == 0:
                unvisited.add(a)
            else:
                u = self.q_table[s][a] + \
                    (C * math.sqrt(math.log(self.N) / self.n_sa[s][a]))
                self.N += 1
                if u > best_u:
                    best_u = u
                    best_a = a
            self.n_sa[s][a] += 1

        if len(unvisited) > 0:
            action = random.sample(unvisited, 1)[0]
        else:
            action = best_a
        return action

    def _get_best_action(self, state, q_table):
        best_q = float('-inf')
        best_a = None
        for action in ROBOT_ACTIONS:
            this_q = q_table[state][action]
            if this_q is not None and this_q > best_q:
                best_q = this_q
                best_a = action
        return best_a

    def __get_best_action(self, state):
        best_q = float('-inf')
        best_a = None
        for action in ROBOT_ACTIONS:
            this_q = self.q_table[state][action]
            if this_q is not None and this_q > best_q:
                best_q = this_q
                best_a = action
        return best_a
