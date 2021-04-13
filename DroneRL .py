import numpy as np
import colorama
from colorama import Fore, Style

BOARD_ROWS = 4
BOARD_COLS = 5
BOARD_HEIGHT = 3
WIN_STATE = []
LOSE_STATE = [(1, 1, 0)]
BlOCK_STATES = []
START = (0, 0, 1)
START_STATE = [(0, 0, 1)]
WIN_REWARD = 1
LOSE_REWARD = -1
DETERMINISTIC = False

RIGHT = "right"
LEFT = "left"
UP = "up"
DOWN = "down"
TOP = "top"
BOTTOM = "bottom"

class Env:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS, BOARD_HEIGHT])
        self.state = state
        self.determine = DETERMINISTIC
        
    def give_reward(self):
        if self.state in WIN_STATE:
            return WIN_REWARD
        elif self.state in LOSE_STATE:
            return LOSE_REWARD
        else:
            return 0

    def is_end(self):
        if (self.state in WIN_STATE) or (self.state in LOSE_STATE):
            return True
        return False

    def _chooseActionProb(self, action):
        if action == "up":
            return np.random.choice(["up", "down", "left", "right", "top", "bottom"], p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06])
        if action == "down":
            return np.random.choice(["down", "up", "left", "right", "top", "bottom"], p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06])
        if action == "left":
            return np.random.choice(["left", "right", "up", "down", "top", "bottom"], p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06])
        if action == "right":
            return np.random.choice(["right", "left", "up", "down", "top", "bottom"], p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06])
        if action == "top":
            return np.random.choice(["top", "bottom", "up", "down", "left", "right"], p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06])
        if action == "bottom":
            return np.random.choice(["bottom", "top", "up", "down", "left", "right"], p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06])

    def next_position(self, action):
        if self.determine:
            if action == UP:
                next_state = (self.state[0], self.state[1] + 1, self.state[2])
            elif action == DOWN:
                next_state = (self.state[0], self.state[1] - 1, self.state[2])
            elif action == LEFT:
                next_state = (self.state[0] - 1, self.state[1], self.state[2])
            elif action == RIGHT:
                next_state = (self.state[0] + 1, self.state[1], self.state[2])
            elif action == TOP:
                next_state = (self.state[0], self.state[1], self.state[2] + 1)
            else:
                next_state = (self.state[0], self.state[1], self.state[2] - 1)
            self.determine = False
        else:
            # non-deterministic
            action = self._chooseActionProb(action)
            self.determine = True
            next_state = self.next_position(action)
        
        # if next state is included in space
        if (next_state[0] >= 0) and (next_state[0] < BOARD_ROWS):
            if (next_state[1] >= 0) and (next_state[1] < BOARD_COLS):
                if (next_state[2] >= 0) and (next_state[2] < BOARD_HEIGHT):
                    return next_state
        return self.state
    

def get_num(action):
    if action == UP:
        return 0
    if action == DOWN:
        return 1
    if action == LEFT:
        return 2
    if action == RIGHT:
        return 3
    if action == TOP:
        return 4
    if action == BOTTOM:
        return 5


class Agent:
    def __init__(self):
        self.states = []
        #self.actions = ["up", "down", "left", "right", "top", "bot"]
        self.actions = [UP, DOWN, LEFT, RIGHT, TOP, BOTTOM]
        self.Env = Env()
        self.lr = 0.2
        self.exp_rate = 0.8
        self.discount = 0.9
        self.target = 0
        self.q_values = np.zeros([len(self.actions), BOARD_HEIGHT, BOARD_COLS, BOARD_ROWS])
        self.policy = np.empty([BOARD_HEIGHT, BOARD_COLS, BOARD_ROWS], dtype=object)
        


    def choose_action(self):
        # choose action with most expected value
        mx_nxt_reward = -200
        action = ""
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                (i, j, k) = self.Env.state
                nxt_reward = self.q_values[get_num(a)][k][j][i]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action
    
        
    def take_action(self, action):
        position = self.Env.next_position(action)
        return Env(state=position)

    def reset(self):
        self.states = []
        self.Env = Env()

    def play(self, rounds=10):
        cnt = 0
        while cnt < rounds:
            self.target = -200
            (i, j, k) = self.Env.state
            action = self.choose_action()
            a = get_num(action)
            self.states.append(self.Env.next_position(action))
            self.Env = self.take_action(action)
            if self.Env.is_end():
                self.target = self.Env.give_reward()
                for l in self.actions:
                    (s, t, r) = self.Env.state
                    if (s, t, r) == WIN_STATE:
                        self.q_values[get_num(l)][r][t][s] = WIN_REWARD
                    else:
                        self.q_values[get_num(l)][r][t][s] = LOSE_REWARD
                self.Env.state = START
            else:
                for l in self.actions:
                    (s, t, r) = self.Env.state
                    tmp = self.Env.give_reward() + self.discount * (self.q_values[get_num(l)][r][t][s])
                    if tmp > self.target:
                        self.target = tmp
            self.q_values[a][k][j][i] = round((1 - self.lr) * self.q_values[a][k][j][i] + self.lr * self.target, 3)
            cnt += 1
            
    def show_values(self):
        for k in range(0, BOARD_HEIGHT):
            print('--------Level  {} --------'.format(k))
            for j in range(0, BOARD_COLS):
                print('----------------------------------')
                out = '| '
                for i in range(0, BOARD_ROWS):
                    max = -200
                    action = ''
                    if (i, BOARD_COLS - j - 1, k) in WIN_STATE:
                        action = 'GOAL'
                        max = 2.0
                    elif (i, BOARD_COLS - j - 1, k) in LOSE_STATE:
                        action = 'LOSS'
                        max = 2.0
                    elif (i, BOARD_COLS - j - 1, k) in START_STATE:
                        action = 'START'
                        max = 2.0
                    for a in self.actions:
                        if max < self.q_values[get_num(a)][k][BOARD_COLS - j - 1][i]:
                            max = self.q_values[get_num(a)][k][BOARD_COLS - j - 1][i]
                            action = a
                    #self.policy[k][BOARD_COLS - j - 1][i] = action
                    out += str(action).ljust(6) + ' | '
                    #out += str(action).ljust(6) + ' | '
                    #out += str(max).ljust(6) + ' | '
                print(out)
            print('----------------------------------')
            print()
            
    def show_values2(self):
        for k in range(0, BOARD_HEIGHT):
            print('--------Q-values - Level {} --------'.format(k))
            for j in range(0, BOARD_COLS):
                print('----------------------------------')
                out = '| '
                for i in range(0, BOARD_ROWS):
                    max = -500
                    action = ''
                    for a in self.actions:
                        if max < self.q_values[get_num(a)][k][BOARD_COLS - j - 1][i]:
                            max = self.q_values[get_num(a)][k][BOARD_COLS - j - 1][i]
                            action = a
                    #self.policy[k][BOARD_COLS - j - 1][i] = action
                    
                    out += str(max).ljust(6) + ' | '
                print(out)
            print('----------------------------------')
            print()
            
            
#just to show the all q value with repect to the all actions
    def show_values3(self):
        for k in range(0, BOARD_HEIGHT):
            print('--------  {} --------'.format(k))
            for j in range(0, BOARD_COLS):
                print('----------------------------------')
                out = '| '
                for i in range(0, BOARD_ROWS):
                    max = -500
                    action = ''
                    if (i, BOARD_COLS - j - 1, k) in WIN_STATE:
                        action = 'GOAL'
                        max = 2.0
                    elif (i, BOARD_COLS - j - 1, k) in LOSE_STATE:
                        action = 'LOSS'
                        max = 2.0
                    elif (i, BOARD_COLS - j - 1, k) in START_STATE:
                        action = 'START'
                        max = 2.0
                        
                    q_values=[]
                    for a in self.actions:
                        q_values.append(str(a) + " : "+ str(self.q_values[get_num(a)][k][BOARD_COLS - j - 1][i]))
                        if max < self.q_values[get_num(a)][k][BOARD_COLS - j - 1][i]:
                            max = self.q_values[get_num(a)][k][BOARD_COLS - j - 1][i]
                            action = a
                    self.policy[k][BOARD_COLS - j - 1][i] = action

                    out += str(q_values).ljust(6) + ' | '
                print(out)
            print('----------------------------------')
            print()



if __name__ == "__main__":
    print(Fore.BLUE +"give the Goal state")
    x = input().split(" ")
    y = (int(x[0]), int(x[1]), int(x[2]))
    WIN_STATE.append(y)
    ag = Agent()
    print("Number of iterations = 10000")
    ag.play(10000)
    ag.show_values()
    ag.show_values2()
    #ag.show_values3() just to show the all q value with repect to the actions
    #print("Number of iteration = 100000")
    #ag.play(100000)
    #ag.show_values()
  


