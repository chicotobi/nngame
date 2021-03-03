import numpy as np
import matplotlib.pyplot as plt

s0 = tuple(tuple(0 for i in range(6)) for i in range(7))

def plot_state(s):
  plt.imshow(np.transpose(s),origin="lower")
  plt.clim(-1,1)
  plt.show()

def step(s,a,val):
  x = a
  y = 0
  while True:
    if s[x][y]==0:
      break
    y += 1
  s = list(s)
  s = [list(i) for i in s]
  s[x][y] = val
  s = [tuple(i) for i in s]
  s = tuple(s)
  
  # Check if player wins vertical
  if y > 2:
      if s[x][y-3] == val and s[x][y-2] == val and s[x][y-1] == val:
        return s, 1
      
  # Check if player wins horizontal
  if s[3][y] == val:
    if s[0][y] == val and s[1][y] == val and s[2][y] == val :
      return s,1
    if s[1][y] == val and s[2][y] == val and s[4][y] == val :
      return s,1
    if s[2][y] == val and s[4][y] == val and s[5][y] == val :
      return s,1
    if s[4][y] == val and s[5][y] == val and s[6][y] == val :
      return s,1
    
  
  # Check if player wins diagonal, three upwards right  
  if x < 4 and y < 3:
    if s[x+1][y+1] == val and s[x+2][y+2] == val and s[x+3][y+3] == val:
      return s,1
    
  # Check if player wins diagonal, two upwards right  
  if 0 < x < 5 and 0 < y < 4:
    if s[x-1][y-1] == val and s[x+1][y+1] == val and s[x+2][y+2] == val:
      return s,1
    
  # Check if player wins diagonal, one upwards right  
  if 1 < x < 6 and 1 < y < 5:
    if s[x-2][y-2] == val and s[x-1][y-1] == val and s[x+1][y+1] == val:
      return s,1
    
  # Check if player wins diagonal, zero upwards right  
  if 2 < x and 2 < y:
    if s[x-3][y-3] == val and s[x-2][y-2] == val and s[x-1][y-1] == val:
      return s,1
    
    
  # Check if player wins diagonal, three downwards right  
  if x < 4 and 2 < y:
    if s[x+1][y-1] == val and s[x+2][y-2] == val and s[x+3][y-3] == val:
      return s,1
    
  # Check if player wins diagonal, two downwards right  
  if 0 < x < 5 and 1 < y < 5:
    if s[x-1][y+1] == val and s[x+1][y-1] == val and s[x+2][y-2] == val:
      return s,1
    
  # Check if player wins diagonal, one downwards right  
  if 1 < x < 6 and 0 < y < 4:
    if s[x-2][y+2] == val and s[x-1][y+1] == val and s[x+1][y-1] == val:
      return s,1
    
  # Check if player wins diagonal, zero downwards right  
  if 2 < x and y < 3:
    if s[x-3][y+3] == val and s[x-2][y+2] == val and s[x-1][y+1] == val:
      return s,1
      
  return s,0  

def get_valid_actions(s):
  if s not in valid_actions:
    valid_actions[s] = [i for i in range(7) if not s[i][5]]
  return valid_actions[s]

def get_policy_agent(s):
  if s in policy and np.random.rand() > eps:
    return policy[s]
  return np.random.choice(get_valid_actions(s))

def get_policy_agent_det(s):
  if s in policy:
    return policy[s]
  if player_mode:
    print("dont know that one, have to choose random")
  return np.random.choice(get_valid_actions(s))

def get_policy_enemy_smart(s):
  if s in old_policy and np.random.rand() > eps:
    return old_policy[s]
  return np.random.choice(get_valid_actions(s))

def get_policy_random(s):
  return np.random.choice(get_valid_actions(s))

def get_policy_person(s):
  return int(input("Action?"))
  
def get_q(s,a):
  if (s,a) not in q:
    q[(s,a)] = 0
  return q[(s,a)]

def add_to_q(s,a,value):
  q[(s,a)] += value

q = {}
old_policy = {}
save = {}
policy = {}
valid_actions = {}

alpha = 0.5
gamma = 1
eps = 0.1

episodes = 500000
results = []
lens = []

player_mode = False

test_each_episodes = 5000
test_episodes = 200
test_results = []
for i in range(episodes):

  # Every fifth episode is against a random enemy to avoid overfitting to itself
  if i%2==0:
    get_policy_enemy = get_policy_random
  else:
    get_policy_enemy = get_policy_enemy_smart    
  
  # Half of the games we play as player 2
  if i%2==0:
    s = s0
    steps = 21
  else:
    a = get_policy_enemy(s0)
    s,r = step(s0,a,1)   
    steps = 20
    
  for j in range(steps):
    # Agent step
    a = get_policy_agent(s)
    s_prime,r = step(s,a,1)
    if r==0:    
      # Enemy step
      a_enemy = get_policy_enemy(s_prime)
      s_prime,r = step(s_prime,a_enemy,-1)
      r *= -1
    
    # Q-learn
    max_value = - np.Infinity
    for a_prime in get_valid_actions(s_prime):
      max_value = max(max_value, get_q(s_prime,a_prime))
    val = alpha * (r + max_value - get_q(s,a))
    add_to_q(s,a,val)   
    
    # Policy improvement
    best_action = None
    max_value = 0
    for a0 in get_valid_actions(s):
      if get_q(s,a0) > max_value:
        best_action = a0
        max_value = get_q(s,a0)
    if best_action:
      policy[s] = best_action    
    if r:
      break
    s = s_prime
  results += [r]
  lens += [j]
  if i%test_each_episodes==0:
    # Evaluate on some games against a random enemy:
    test_result = 0
    for j in range(test_episodes):
    
      # Half of the games we play as player 2
      if j%2==0:
        s = s0
        steps = 21
      else:
        a = get_policy_random(s0)
        s,r = step(s0,a,1)   
        steps = 20
      
      for k in range(steps):
        # Agent step
        a = get_policy_agent_det(s)
        s_prime,r = step(s,a,1)
        if r==0:    
          # Enemy step
          a_enemy = get_policy_random(s_prime)
          s_prime,r = step(s_prime,a_enemy,-1)
          r *= -1
        if r:
          break
        s = s_prime
      test_result += r/test_episodes
    
    # Update enemy policy
    old_policy = save.copy()
    save = policy.copy()
    print("I have a decision for ",len(policy)," states.")
    print("Mean length:", np.mean(lens[-500:]))
    print("Mean result:", np.mean(results[-500:]))
    test_results += [test_result]
    plt.plot(test_results)
    plt.show()
    #plot_state(s_prime)

player_mode = True

while True:
  # Play a game against the computer
  s = s0
  steps = 21
  
  for j in range(steps):
  # Agent step
    a = get_policy_agent_det(s)
    s_prime,r = step(s,a,1)
    plot_state(s_prime)
    if r==0:    
      # Enemy step
      a_enemy = get_policy_person(s_prime)
      s_prime,r = step(s_prime,a_enemy,-1)
      r *= -1
    s = s_prime
    plot_state(s_prime)
    if r:
      break