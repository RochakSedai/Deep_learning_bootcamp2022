from numpy.random import random
from random import randint

# epsilon greedy strategy the agent visit new styates with epsilon probability
EPSILON = 0.1

# number of bandit
BANDITS = 3
# number of iterations
EPISODES = 10000

class Bandit:

    def __init__(self, probability):
        # QK(a) stores the mean of rewards
        self.q = 0
        # k means how many times action a (so the bandit) was chosen in the past
        self.k = 0
        # probability distribution
        self.probability = probability

    def get_reward(self):
        # rewards can be +1 (win) or 0 (loss)
        if random() < self.probability:
            return 1
        else: 
            return 0

class NArmedBandit:

    def __init__(self):
        self.bandits = []
        self.bandits.append(Bandit(0.5))            
        self.bandits.append(Bandit(0.6))            
        self.bandits.append(Bandit(0.4))            
    
    def run(self):
        for i in range(EPISODES):
            bandit = self.bandits[self.select_bandit()]
            reward = bandit.get_reward()
            self.update(bandit, reward)
            print('Iteration %s, bandit %s with Q value %s' %(i, bandit.probability, bandit.q))

    def select_bandit(self):
        # this is the epsilon greedy strategy
        # with epsiolon probability the agent explore - otherwise it exploits
        if random() < EPSILON:
            # exploration
            bandit_index = randint(0, BANDITS - 1)
        else: 
            #exploitation
            bandit_index = self.get_bandit_max_q()

        return bandit_index

    def update(self, bandit, reward):
        bandit.k = bandit.k + 1
        bandit.q = bandit.q + (1/ (1 + bandit.k)) * (reward - bandit.q)

    def get_bandit_max_q(self):
        # we find the bandit with max Q(a) value for the greedy exploitation
        # we need the index of the bandit with max Q(a)
        max_q_bandit_index = 0
        max_q = self.bandits[max_q_bandit_index].q

        for i in range(1, BANDITS):
            if self.bandits[i].q > max_q:
                max_q = self.bandits[i].q
                max_q_bandit_index = i

        return max_q_bandit_index

    # show statistics: how many times the given bandit was chosen
    def show_statistics(self):
        for i in range(BANDITS):
            print('Bandit %s with k: %s' % (i, self.bandits[i].k))

if __name__ == '__main__':
    bandit_problem = NArmedBandit()
    bandit_problem.run()
    bandit_problem.show_statistics()