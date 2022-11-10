import self
try:
    file1 = open("noe", "r")
    file = open("alpha01.txt", "r")
    self.q_rewards1 = file.read()
    file.close()
    file = open("alpha001.txt", "r")
    self.q_rewards2 = file.read()
    file.close()
    file = open("alpha0001.txt", "r")
    self.q_rewards3 = file.read()
    file.close()
except:
    print("starter på alpha 0.1")
    for ep in range(self.n_test_episodes):
        s = self.environment.get_init_state()  # initialize state
        episode_reward1 = 0
        for t in range(self.max_timestep_per_episode):
            a = self.epsilon_greedy(s)
            r, s_next = self.environment._Environment__apply_dynamics(
                s, a)
            episode_reward1 += r
            # Update q-value for the (state, action) pair
            old_q1 = self.q_table1[s][a]
            best_next1 = self._get_best_action(s_next, self.q_table1)
            best_next_q1 = self.q_table1[s_next][best_next1]
            if self.environment.is_solved(s_next):
                best_next_q1 = 0
            target1 = r + self.gamma * best_next_q1
            new_q1 = old_q1 + self.alpha1 * (target1 - old_q1)
            self.q_table1[s][a] = new_q1
            s = s_next
            if self.environment.is_solved(s):
                break
        length = len(self.q_rewards1)
        if length >= 50:
            self.q_rewards1.append(
                (sum(self.q_rewards1[-50:-1]) + episode_reward1)/50)
        else:
            self.q_rewards1.append((
                sum(self.q_rewards1[-length:-1]) + episode_reward1)/(length+1))
    file = open("alpha01.txt", "w+")
    content = str(self.q_rewards1)
    file.write(content)
    file.close()
    print("starter på alpha 0.01")
    for ep in range(self.n_test_episodes):
        s = self.environment.get_init_state()  # initialize state
        episode_reward2 = 0
        for t in range(self.max_timestep_per_episode):
            a = self.epsilon_greedy(s)
            r, s_next = self.environment._Environment__apply_dynamics(
                s, a)
            episode_reward2 += r
            # Update q-value for the (state, action) pair
            old_q2 = self.q_table2[s][a]
            best_next2 = self._get_best_action(s_next, self.q_table2)
            best_next_q2 = self.q_table2[s_next][best_next2]
            if self.environment.is_solved(s_next):
                best_next_q2 = 0
            target2 = r + self.gamma * best_next_q2
            new_q2 = old_q2 + self.alpha2 * (target2 - old_q2)
            self.q_table2[s][a] = new_q2
            s = s_next
            if self.environment.is_solved(s):
                break
        length = len(self.q_rewards2)
        if length >= 50:
            self.q_rewards2.append(
                (sum(self.q_rewards2[-50:-1]) + episode_reward2)/50)
        else:
            self.q_rewards2.append((
                sum(self.q_rewards2[-length:-1]) + episode_reward2)/(length+1))
    file = open("alpha001.txt", "w+")
    content = str(self.q_rewards2)
    file.write(content)
    file.close()
    print("starter på alpha 0.001")
    for ep in range(self.n_test_episodes):
        s = self.environment.get_init_state()
        episode_reward3 = 0
        for t in range(self.max_timestep_per_episode):
            a = self.epsilon_greedy(s)
            r, s_next = self.environment._Environment__apply_dynamics(
                s, a)
            episode_reward3 += r
            # Update q-value for the (state, action) pair
            old_q3 = self.q_table3[s][a]
            best_next3 = self._get_best_action(s_next, self.q_table3)
            best_next_q3 = self.q_table3[s_next][best_next3]
            if self.environment.is_solved(s_next):
                best_next_q3 = 0
            target3 = r + self.gamma * best_next_q3
            new_q3 = old_q3 + self.alpha3 * (target3 - old_q3)
            self.q_table3[s][a] = new_q3
            s = s_next
            if self.environment.is_solved(s):
                break
        length = len(self.q_rewards3)
        if length >= 50:
            self.q_rewards3.append(
                (sum(self.q_rewards3[-50:-1]) + episode_reward3)/50)
        else:
            self.q_rewards3.append((
                sum(self.q_rewards3[-length:-1]) + episode_reward3)/(length+1))
    file = open("alpha0001.txt", "w+")
    content = str(self.q_rewards3)
    file.write(content)
    file.close()


# if self.show_graph:
    #     file = open("q_rewards.txt", "r")
    #     self.q_rewards = file.read().split(",")
    #     file.close()

    #     file = open("SARSA_rewards.txt", "r")
    #     self.sarsa_rewards = file.read().split(",")
    #     file.close()

    #     plt.style.use('ggplot')
    #     time = [x for x in range(len(self.sarsa_rewards))]
    #     print(self.q_rewards)
    #     print(self.sarsa_rewards)
    #     plt.plot(time, self.q_rewards, label='Q-learning')
    #     plt.plot(time, self.sarsa_rewards, label='SARSA')
    #     plt.ylabel('50-step moving average reward')
    #     plt.xlabel('Episode')
    #     plt.title(
    #         '50-step moving average reward received by agent with different learning rate')
    #     plt.legend(loc='upper left')
    #     plt.show()
    #     self.show_graph = False
