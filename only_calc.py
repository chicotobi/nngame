import scipy.optimize
from math import sin,cos
import numpy as np
import matplotlib.pyplot as plt

np.seterr(over="raise")

def linear_control_policy(c_theta,c_omega,c_v,c_x,x_goal):
  return lambda tt,oo,vv,xx: c_theta*tt+c_omega*oo+c_v*vv + c_x*(xx-x_goal)
  
def update_state(F,theta,omega,v,x,y,dt):
  m = 1
  M = 1
  l = 2
  
  g = 9.81

  t1 = (M+m)*l/cos(theta)
  t2 = -m*l*cos(theta)
  
  f1 = (M+m)*g*sin(theta)/cos(theta)
  f2 = -m*l*omega**2*sin(theta)
  
  alpha = (F+f1+f2)/(t1+t2)
  a = (l*alpha - g*sin(theta))/cos(theta)
  
  omega += dt*alpha
  theta += dt*omega
  v += dt*a
  x += dt*v
  
  x1 = x - sin(theta)*l
  y1 = y - cos(theta)*l
  
  return [theta,omega,v,x,y,x1,y1]

def get_score(x1,y1,x_goal,y_goal,dt):  
  dist = ((x1-x_goal)**2+(y1-y_goal)**2)**.5  
  return -dt*dist

def play(pol,x_goal,y_goal):
  score = 10
  t = 0
  dt = 0.01
  
  omega = 0
  theta = 0
  
  v = 0
  x = 2
  y = 3
  
  arr_score = []
    
  while t<10:
    f = pol(theta,omega,v,x)
    [theta,omega,v,x,y,x1,y1] = update_state(f,theta,omega,v,x,y,dt)  
    score += get_score(x1,y1,x_goal,y_goal,dt)
    arr_score.append(score)
    t += dt
  return arr_score

def game(para):
  c_theta, c_omega, c_v, c_x = para
  
  x_goal = 3  
  y_goal = 1
  p = linear_control_policy(c_theta,c_omega,c_v,c_x,x_goal)
  arr_score = play(p,x_goal,y_goal)
  sc = arr_score[-1]
  return -sc

ranges = [(-110,-90),(-55,-45),(8,12),(1,10)]
sol = scipy.optimize.brute(game, ranges=ranges,Ns=5, finish=None,full_output=True)

for p in np.linspace(1,10,20): 
  p = linear_control_policy(-100,-50,10,p,3)
  arr_scores = play(p,3,1)
  plt.subplot(2,1,1)
  plt.plot(arr_scores)
  plt.subplot(2,1,2)
  plt.plot(np.diff(arr_scores))