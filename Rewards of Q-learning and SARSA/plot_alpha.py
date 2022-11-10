import matplotlib.pyplot as plt
from ast import literal_eval
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))

file = open("alpha01.txt", "r")
alpha01 = literal_eval(file.read())
file.close()

file = open("alpha001.txt", "r")
alpha001 = literal_eval(file.read())
file.close()

file = open("alpha0001.txt", "r")
alpha0001 = literal_eval(file.read())
file.close()

alpha01 = alpha01[10:]
alpha001 = alpha001[10:]
alpha0001 = alpha0001[10:]

plt.style.use('ggplot')
time = [x for x in range(1800)]
plt.plot(alpha0001, label='Alpha = 0.001')
plt.plot(alpha001, label='Alpha = 0.01')
plt.plot(alpha01, label='Alpha = 0.1')
plt.ylabel('50-step moving average reward')
plt.xlabel('Episode')
plt.title(
    '50-step moving average reward received by different alpha')
plt.legend(loc='lower right')
plt.show()
