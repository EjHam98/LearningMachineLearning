import numpy as np
import random
import math


class Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.map = []
        for i in range(self.height):
            self.map.append([])
            for j in range(self.width):
                self.map[i].append(0)

    def obstacles(self, walls):
        if len(walls) != self.height or len(walls[0]) != self.width:
            return
        for i in range(self.height):
            for j in range(self.width):
                self.map[i][j] = walls[i][j]

    def x(self, pos):
        return pos % self.width

    def y(self, pos):
        return pos // self.width

    def distance(self, pos1, pos2):
        return (self.x(pos1) - self.x(pos2)) + (self.y(pos1) - self.y(pos2))

    def invalid(self, pos, direction):
        res = False
        if direction == 0:
            if self.y(pos) == 0 or self.map[self.y(pos) - 1][self.x(pos)] != 0:
                res = True
        if direction == 1:
            if self.x(pos) == self.width - 1 or self.map[self.y(pos)][self.x(pos) + 1] != 0:
                res = True
        if direction == 2:
            if self.y(pos) == self.height - 1 or self.map[self.y(pos) + 1][self.x(pos)] != 0:
                res = True
        if direction == 3:
            if self.x(pos) == 0 or self.map[self.y(pos)][self.x(pos) - 1] != 0:
                res = True
        return res
    def draw_text(self, pos, target):
        for i in range(self.height):
            for j in range(self.width):
                if i == self.y(pos) and j == self.x(pos):
                    print("8", end=" ")
                elif i == self.y(target) and j == self.x(target):
                    print("$", end=" ")
                elif self.map[i][j] == 0:
                    print("+", end=" ")
                else:
                    print("#", end=" ")
            print("")
        print("=========================\n")


class QLearning:
    def __init__(self, map2d, x, y):
        self.map2D = map2d
        if x < 1 or x > self.map2D.height:
            x = 1
        if y < 1 or y > self.map2D.width:
            y = 1
        x -= 1
        y -= 1
        self.original_state = self.state = y+x*self.map2D.width
        self.states = self.map2D.height*self.map2D.width
        self.actions = 5
        self.action = 0
        self.new_state = 0
        self.target = self.map2D.height*self.map2D.width-1
        self.memory = np.zeros((self.states, self.actions))
        self.rand_factor = 0.2  # epsilon
        self.gamma = 0.85
        self.learn_rate = 0.75
        self.iterations = self.map2D.height*self.map2D.width
        self.rep = []
        for i in range(self.map2D.height):
            self.rep.append([])
            for j in range(self.map2D.width):
                self.rep[i].append(0)

    def set_target(self, x, y):
        x -= 1
        y -= 1
        self.target = y + x * self.map2D.width
        
    def reward(self, old_pos, new_pos):
        if new_pos == self.target:
            return 2000
        if old_pos == new_pos:
            return -100
        rew = (self.map2D.distance(old_pos, self.target) - self.map2D.distance(new_pos, self.target))
        if self.rep[self.map2D.y(self.state)][self.map2D.x(self.state)] > 0 and rew > 0:
            rew = -0.1*rew*self.rep[self.map2D.y(self.state)][self.map2D.x(self.state)]
        return rew

    def apply_action(self, act, old_state):
        if act == 0:
            return old_state - self.map2D.width
        if act == 1:
            return old_state + 1
        if act == 2:
            return old_state + self.map2D.width
        if act == 3:
            return old_state - 1
        if act == 4:
            return old_state

    #One iteration of exploration
    def fit_map(self):
        for i in range(self.map2D.height):
            for j in range(self.map2D.width):
                self.rep[i][j] = 0
        for iteration in range(self.iterations):
            if random.uniform(0, 1) <= self.rand_factor:
                self.action = random.randint(0, self.actions - 1)
            else:
                maxi = k = 0
                for i in range(0, self.actions - 1):
                    if self.memory[self.state, i] >= maxi:
                        k = i
                        maxi = self.memory[self.state, k]
                self.action = k
            if self.map2D.invalid(self.state, self.action):
                self.memory[self.state, self.action] = -1000
                continue
            self.new_state = self.apply_action(self.action, self.state)
            self.memory[self.state, self.action] += int(self.learn_rate *
                        (self.reward(self.state, self.new_state) +
                         self.gamma * np.max(self.memory[self.new_state, :]) - self.memory[self.state, self.action]))
            self.state = self.new_state
            self.rep[self.map2D.y(self.state)][self.map2D.x(self.state)] += 1
            """if self.state == target:
                self.state = rows*columns-1
                epsilon -= 0.05"""
        self.state = self.original_state

    #Print out the found solution
    def best_route(self):
        u = 0
        while(u<100):
            u += 1
            self.map2D.draw_text(self.state, self.target)
            if self.state == self.target:
                break
            maxi = k = 0
            for i in range(0, self.actions):
                if self.memory[self.state, i] >= maxi:
                    k = i
                    maxi = self.memory[self.state, k]
            self.action = k
            if not self.map2D.invalid(self.state, self.action):
                self.state = self.apply_action(self.action, self.state)


def main():
    walls = [[0, 0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0, 1, 0],
             [1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]]

    mymap = Map(7, 7)
    mymap.obstacles(walls)

    ai = QLearning(mymap, 1, 1)
    ai.set_target(7, 1)
    ai.rand_factor = 0.8
    # ai.learn_rate = 0.5
    for i in range(500):
        ai.fit_map()
    ai.best_route()



if __name__ == "__main__":
    main()
