import numpy as np

from rl.policy import Policy



class CustomEpsGreedy(Policy):
    def __init__(self, max_eps = 1., min_eps = 0.1, eps_decay=0.9997):
        super(CustomEpsGreedy, self).__init__
        self.eps = max_eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)

        if self.eps > self.min_eps:
            self.eps = self.eps*self.eps_decay
        else:
            self.eps = self.min_eps

        return action

