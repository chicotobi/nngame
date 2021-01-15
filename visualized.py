import tkinter as tk
from math import sin,cos

t = 0
m = 1
M = 1
l = 2

dt = 0.01
g = 9.8

alpha = 0
omega = 0
theta = 0

a = 0
v = 0
x = 2
y = 3

x_goal = 3
y_goal = 3

score = 0

class Car(object):
    def __init__(self, canvas, *args, **kwargs):
        self.canvas = canvas
        
        x1 = x - sin(theta)*l
        y1 = cos(theta)*l
        self.id = canvas.create_oval(x*100-10,y*100-10,x*100+10,y*100+10)
        self.pendulum = canvas.create_oval(x1*100-5,y1*100-5,x1*100+5,y1*100+5)
        self.line = canvas.create_line(x*100,y*100,x1*100,y1*100)
        self.goal = canvas.create_rectangle(x_goal*100-10,y_goal*100-10,x_goal*100+10,y_goal*100+10)
        
    def setAccelerations(self,F):
      global t,dt,alpha,omega,theta,a,v,x
      t1 = (M+m)*l/cos(theta)
      t2 = -m*l*cos(theta)
      
      f1 = (M+m)*g*sin(theta)/cos(theta)
      f2 = -m*l*omega**2*sin(theta)
      
      alpha = (F+f1+f2)/(t1+t2)
      a = (l*alpha - g*sin(theta))/cos(theta)

    def update_state(self):
        global t,dt,alpha,omega,theta,a,v,x
        t += dt
        omega += dt*alpha
        theta += dt*omega
        v += dt*a
        x += dt*v

    def move(self):
        global t,dt,alpha,omega,theta,a,v,x,y       
        F_control = -theta*100 - omega*50 + v*10 + (x-x_goal)*3.0
        self.update_state()
        self.setAccelerations(F_control)
        
        x1 = x - sin(theta)*l
        y1 = y - cos(theta)*l
        self.canvas.coords(self.id,x*100-10,y*100-10,x*100+10,y*100+10)
        self.canvas.coords(self.pendulum,x1*100-5,y1*100-5,x1*100+5,y1*100+5)
        self.canvas.coords(self.line,x*100,y*100,x1*100,y1*100)        
        self.canvas.coords(self.goal,x_goal*100-10,y_goal*100-10,x_goal*100+10,y_goal*100+10)
        
class App(object):
    def __init__(self, master, **kwargs):
        self.master = master
        self.canvas = tk.Canvas(self.master, width=600, height=600)
        self.canvas.pack()
        self.car = Car(self.canvas, 2, 2, 12, 12)
        self.canvas.pack()
        self.master.bind('<Left>', self.leftKey)
        self.master.bind('<Right>', self.rightKey)
        self.master.bind('<Up>', self.upKey)
        self.master.bind('<Down>', self.downKey)
        self.master.after(0, self.animation)

    def leftKey(self,event):
        global x_goal
        x_goal -= 0.1
        
    def rightKey(self,event):
        global x_goal
        x_goal += 0.1
        
    def upKey(self,event):
        self.car.ay = -1
        
    def downKey(self,event):
        self.car.ay = 1
  
    def animation(self):
        self.car.move()
        self.master.after(int(dt*1000), self.animation)

root = tk.Tk()
app = App(root)
root.mainloop()