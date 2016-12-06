import numpy as np

EPSILON = 10e-2


def distance(lhs, rhs):
    return np.linalg.norm(lhs - rhs)

def equal(lhs, rhs):
    return distance(lhs, rhs) < EPSILON

def sample():
    MAX = 100
    return np.random.rand(2) * 2 * MAX - MAX

class Node:

    def __init__(self, parent, q):
        self.parent = parent
        self.q = q

class Tree:

    def __init__(self, q):
        self.nodes = [Node(None, q)]

    def nearest_neighbor(self, q):
        index = len(self.nodes)
        min_distance = float("inf")
        for i in range(len(self.nodes)):
            d = distance(q, self.nodes[i].q)
            if d < min_distance:
                index = i
                min_distance = d
        return self.nodes[index]


def new_config(q, q_near):
    d = distance(q, q_near)
    q_new = q_near + (q - q_near) * EPSILON / d
    return q_new


REACHED = 0
ADVANCED = 1
TRAPPED = 2


def extent(t, q):
    node_near = t.nearest_neighbor(q)
    q_new = new_config(q, node_near.q)

    if not q_new is None:
        t.nodes.append(Node(node_near, q_new))
        if equal(q, q_new):
            return REACHED
        else:
            return ADVANCED
    return TRAPPED


def build_rrt(q_init, q_goal, max_iteration):
    t = Tree(q_init)
    for i in range(max_iteration):
        q_rand = sample()
        if not extent(t, q_rand) == TRAPPED:
            last_node = t.nodes[-1]
            if distance(q_goal, last_node.q) < EPSILON:
                print "success"
                return t
    return t



q_init = np.array([0, 0])
q_goal = np.array([60, 40])

t = build_rrt(q_init, q_goal, 1000)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for n in t.nodes:
    if not n.parent is None:
        ax.plot([n.parent.q[0], n.q[0]], [n.parent.q[1], n.q[1]], "r-")

ax.plot([q_init[0]], [q_init[1]], "bo")
ax.plot([q_goal[0]], [q_goal[1]], "g^")

plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.show()


