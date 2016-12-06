import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from RRT_Connect import build_rrt_connect

class Point:

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def joint_point(self):
        return Point(self.x, self.y)


class Link:

    def __init__(self, link, r, theta):
        self.previous_link = link
        self.r = r
        self.theta = theta

    def start(self):
        return self.previous_link.joint_point()

    def end(self):
        p = self.start()
        p.x += r * np.cos(self.theta)
        p.y += r * np.sin(self.theta)
        return p

    def joint_point(self):
        return self.end()


fig, ax = plt.subplots()

r = 10
theta1 = 0
theta2 = 0

q_init = np.array([theta1, theta2])
q_goal = np.array([0.75 * np.pi, 0.75 * np.pi])

ground = Point(0, 0)
link1 = Link(ground, r, theta1)
link2 = Link(link1, r, theta2)

start = link1.start()
end = link1.end()
link1_line, = ax.plot([start.x, end.x], [start.y, end.y], 'r-')

start = link2.start()
end = link2.end()
link2_line, = ax.plot([start.x, end.x], [start.y, end.y], 'r-')

a = np.array([-2, 30])
b = np.array([2, 30])
c = np.array([-2, 12])
d = np.array([2, 12])

ax.plot([a[0], b[0]], [a[1], b[1]], 'b-')
ax.plot([b[0], d[0]], [b[1], d[1]], 'b-')
ax.plot([d[0], c[0]], [d[1], c[1]], 'b-')
ax.plot([c[0], a[0]], [c[1], a[1]], 'b-')


t1, t2, qs = build_rrt_connect(q_init, q_goal, 1000)

print qs
print len(qs)
print len(qs[0])
print qs[0][0][0]
# qs = [(x, y) for (x, y) in zip(np.linspace(0, 2 * np.pi, 200), np.linspace(0, -2 * np.pi, 200))]

def update_link(link, line, theta):
    link.theta = theta
    start = link.start()
    end = link.end()
    line.set_xdata([start.x, end.x])
    line.set_ydata([start.y, end.y])
    return line

def animate(i):
    index = i % len(qs)
    theta1 = qs[index][0][0]
    theta2 = qs[index][0][1]
    line1 = update_link(link1, link1_line, theta1)
    line2 = update_link(link2, link2_line, theta2)
    return link1, link2

def init():
    return link1, link2

# ani = animation.FuncAnimation(fig, animate, np.arange(1, len(qs)),
#         init_func=init, interval=2, blit=False)

ani = animation.FuncAnimation(fig, animate, interval=len(qs))



plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.show()
