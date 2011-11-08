"""
An early-stopping heuristic
"""

class early_stopping(object):
    def __init__(self, warmup, improvement_thresh=0.5, patience=2.0):
        self.warmup = warmup
        self.improvement_thresh = improvement_thresh
        self.patience = patience
        self.curtime = 0
        self.best_time = -1
        self.best_y = float('inf')
        self.best_y_std = 0
        self.cur_y = None
        self.cur_y_std = None

    def step(self, y, y_std):
        if y_std < 0:
            raise ValueError('negative y_std', y_std)

        self.curtime += 1
        self.cur_y = y
        self.cur_y_std = y_std

        if y < self.best_y - self.improvement_thresh * self.best_y_std:
            self.best_time = self.curtime
            self.best_y = y
            self.best_y_std = y_std

    def done(self):
        return self.curtime >= max(
            self.warmup,
            self.best_time * self.patience)

