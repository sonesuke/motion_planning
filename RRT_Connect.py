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

# def in_collision(q):
#     if (q[0] - 30)**2 + (q[1] - 30)**2 < 20**2:
#         return True
#     return False

def is_cross(q_a, q_b, q_c, q_d):
    ax = q_a[0]
    ay = q_a[1]
    bx = q_b[0]
    by = q_b[1]
    cx = q_c[0]
    cy = q_c[1]
    dx = q_d[0]
    dy = q_d[1]
    ta = (cx - dx) * (ay - cy) + (cy - dy) * (cx - ax)
    tb = (cx - dx) * (by - cy) + (cy - dy) * (cx - bx)
    tc = (ax - bx) * (cy - ay) + (ay - by) * (ax - cx)
    td = (ax - bx) * (dy - ay) + (ay - by) * (ax - dx)
    return tc * td < 0 and ta * tb < 0

def in_collision(q):
    r = 10
    link1_start = np.array([0, 0])
    link1_end = np.array([r * np.cos(q[0]), r* np.sin(q[0])])
    link2_start = link1_end
    link2_end = link2_start + np.array([r * np.cos(q[1]), r* np.sin(q[1])])

    a = np.array([-2, 30])
    b = np.array([2, 30])
    c = np.array([-2, 12])
    d = np.array([2, 12])

    if is_cross(link1_start, link1_end, a, c):
        return True

    if is_cross(link2_start, link2_end, a, c):
        return True

    if is_cross(link1_start, link1_end, b, d):
        return True

    if is_cross(link2_start, link2_end, b, d):
        return True

    if is_cross(link1_start, link1_end, c, d):
        return True

    if is_cross(link2_start, link2_end, c, d):
        return True


    return False


def new_config(q, q_near):
    d = distance(q, q_near)
    q_new = q_near + (q - q_near) * EPSILON / d
    if in_collision(q_new):
        return None
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


def connect(t, q):
    ret = ADVANCED
    while True:
        ret = extent(t, q)
        if not ret == ADVANCED:
            break
    return ret


def path(t_init, t_goal):
    t_init_last = t_init.nodes[-1]
    t_goal_last = t_goal.nodes[-1]

    ret = []
    n_init_current = None
    n_goal_current = None

    n_init = t_init.nearest_neighbor(t_goal_last.q)
    n_goal = t_goal.nearest_neighbor(t_init_last.q)
    if distance(n_goal.q, t_init_last.q) < EPSILON:
        n_init_current = t_init_last
        n_goal_current = n_goal
    elif distance(n_init.q, t_goal_last.q) < EPSILON:
        n_init_current = n_init
        n_goal_current = t_goal_last

    init_path = []
    while n_init_current.parent is not None:
        init_path.append((n_init_current.parent.q, n_init_current.q))
        n_init_current = n_init_current.parent
    init_path.reverse()

    goal_path = []
    while n_goal_current.parent is not None:
        goal_path.append((n_goal_current.q, n_goal_current.parent.q))
        n_goal_current = n_goal_current.parent

    return init_path + goal_path

def is_no_obstacle(q_start, q_end):
    d = distance(q_start, q_end)
    q_new = q_start + (q_end - q_start) * EPSILON / d

    while distance(q_new, q_end) > EPSILON:
        q_new = q_new + (q_end - q_start) * EPSILON / d
        if in_collision(q_new):
            return False
    return True



def post_process(path, k):
    ret = path

    ite = 0
    while ite < k:
        (i, j) = np.sort(np.random.random_integers(0, len(ret)-1, 2))
        if i == j:
            continue

        if len(ret) < 3:
            break
        ite += 1
        if is_no_obstacle(ret[i][0], ret[j][0]):
            new_ret = []
            for forward in range(i+1):
                new_ret.append(ret[forward])
            for backward in range(j, len(ret)):
                new_ret.append(ret[backward])
            ret = new_ret

    print len(path)
    print len(ret)
    print ret

    interpolate = []
    for i in range(len(ret)-1):
        q_start = ret[i][0]
        q_end = ret[i+1][0]
        d = distance(q_start, q_end)
        q_new = q_start
        while distance(q_new, q_end) > EPSILON:
            q_next = q_new + (q_end - q_start) * EPSILON / d
            interpolate.append((q_new, q_next))
            q_new = q_next

    interpolate.append(ret[-1])
    return interpolate


def build_rrt_connect(q_init, q_goal, max_iteration):
    t_init = Tree(q_init)
    t_goal = Tree(q_goal)

    t_a = t_init
    t_b = t_goal

    for i in range(max_iteration):
        q_rand = sample()
        if not extent(t_a, q_rand) == TRAPPED:
            last_node = t_a.nodes[-1]
            if connect(t_b, last_node.q) == REACHED:
                print "success"
                return t_init, t_goal, path(t_init, t_goal)
            t_tmp = t_a
            t_a = t_b
            t_b = t_tmp

    return t_init, t_goal, []


if __name__ == '__main__':
    q_init = np.array([0, 0])
    q_goal = np.array([80, 40])

    t1, t2, pth = build_rrt_connect(q_init, q_goal, 1000)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for n in t1.nodes:
        if not n.parent is None:
            ax.plot([n.parent.q[0], n.q[0]], [n.parent.q[1], n.q[1]], "r-")

    for n in t2.nodes:
        if not n.parent is None:
            ax.plot([n.parent.q[0], n.q[0]], [n.parent.q[1], n.q[1]], "m-")

    ax.plot([q_init[0]], [q_init[1]], "bo")
    ax.plot([q_goal[0]], [q_goal[1]], "g^")

    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.show()



