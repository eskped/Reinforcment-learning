import matplotlib.pyplot as plt
from ast import literal_eval
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))

file = open("alpha001.txt", "r")
q_rewards = literal_eval(file.read())
file.close()

file = open("SARSA_rewards.txt", "r")
sarsa_rewards = literal_eval(file.read())
file.close()

q_rewards = q_rewards[10:]
sarsa_rewards = sarsa_rewards[10:]

# for i in range(len(sarsa_rewards)):
#     sarsa_rewards[i] -= 1000


plt.style.use('ggplot')
time = [x for x in range(1800)]
plt.plot(q_rewards, label='Q-learning')
plt.plot(sarsa_rewards, label='SARSA')
plt.ylabel('50-step moving average reward')
plt.xlabel('Episode')
plt.title(
    '50-step moving average reward received by Q-learning and SARSA')
plt.legend(loc='upper left')
plt.show()
