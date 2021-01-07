import tkinter as tk
from math import pi,sin,cos
from scipy.integrate import odeint

class Car(object):
    def __init__(self, canvas, *args, **kwargs):
        self.canvas = canvas
        self.x0 = 200
        self.y0 = 200
        self.g = 9.81
        self.l = 80
        self.t = 0
        self.theta = 1
        self.theta_dot = 0
        self.vx = 0
        self.vy = 0
        
        self.ax = 0
        self.ay = 0
        self.calc_pendulum()
        
        self.id = canvas.create_oval(self.x0,self.y0,self.x0+20,self.y0+20)
        self.pendulum = canvas.create_oval(self.x1,self.y1,self.x1+10,self.y1+10)
        self.line = canvas.create_line(self.x0,self.y0,self.x1,self.y1)
        
    def calc_pendulum(self):
        self.t += 1
        
        ax = self.ax
        ay = self.ay
        g = self.g
        l = self.l
        
        def f(u,x):
            f0 = u[1]
            f1 = (-ax * cos(u[0]) - (g + ay) * sin(u[0])) / l
            return (f0,f1)
          
        res = odeint(f, (self.theta, self.theta_dot), [0,0.1])
        
        self.theta = res[-1,0]
        self.theta_dot = res[-1,1]
        
        self.x1 = self.x0 + self.l * sin(self.theta)
        self.y1 = self.y0 + self.l * cos(self.theta)

    def move(self):
        self.vx += 0.1*self.ax
        self.vy += 0.1*self.ay
        self.x0 += self.vx
        self.y0 += self.vy
        self.calc_pendulum()
        self.ax = 0
        self.ay = 0
        self.canvas.coords(self.id,self.x0-10,self.y0-10,self.x0+10,self.y0+10)
        self.canvas.coords(self.pendulum,self.x1-5,self.y1-5,self.x1+5,self.y1+5)
        self.canvas.coords(self.line,self.x0,self.y0,self.x1,self.y1)
        

class App(object):
    def __init__(self, master, **kwargs):
        self.master = master
        self.canvas = tk.Canvas(self.master, width=400, height=400)
        self.canvas.pack()
        self.car = Car(self.canvas, 2, 2, 12, 12)
        self.canvas.pack()
        self.master.bind('<Left>', self.leftKey)
        self.master.bind('<Right>', self.rightKey)
        self.master.bind('<Up>', self.upKey)
        self.master.bind('<Down>', self.downKey)
        self.master.after(0, self.animation)

    def leftKey(self,event):
        self.car.ax = -1
        
    def rightKey(self,event):
        self.car.ax = 1
        
    def upKey(self,event):
        self.car.ay = -1
        
    def downKey(self,event):
        self.car.ay = 1
  
    def animation(self):
        self.car.move()
        self.master.after(50, self.animation)

root = tk.Tk()
app = App(root)
root.mainloop()