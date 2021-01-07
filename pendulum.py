import tkinter

main = tkinter.Tk()

def leftKey(event):
    print("Left key pressed")

def rightKey(event):
    print("Right key pressed")

w = tkinter.Canvas(main, width=500, height=500)
main.bind('<Left>', leftKey)
main.bind('<Right>', rightKey)
w.pack()
w.create_oval(50, 50, 60, 60)
main.mainloop()


