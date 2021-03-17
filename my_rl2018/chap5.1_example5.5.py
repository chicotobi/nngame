import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt

npr.seed(0)

n_trials = 10
n_episodes = int(1e7)
ordinary = np.zeros((n_episodes,n_trials))
for trial in range(n_trials):
  print(trial)
  my_sum = 0
  n_states = 0
  n_visits = 1
  for i in range(n_episodes):
    rhor = 1
    while True:
      if npr.rand()<.5:
        rhor = 0
        break
      rhor *= 2
      if npr.rand()>0.9:
        break   
      n_visits += 1
    my_sum += rhor
    ordinary[i,trial] = my_sum / n_visits
plt.semilogx(range(1,n_episodes+1),ordinary)
plt.ylim(0,2)