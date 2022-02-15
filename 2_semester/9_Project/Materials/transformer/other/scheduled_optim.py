from numpy import power, min


class ScheduledOptim(object):
    def __init__(self, h_s, n_warmup_steps, mul=1):
        self.h_s = h_s
        self.n_warmup = n_warmup_steps
        self.steps = 0

        self.mul = mul

    def update_learning_rate(self, optimizer):
        self.steps += 1

        lr = (
            self.mul
            * power(self.h_s, -0.5)
            * min([power(self.steps, -0.5), power(self.n_warmup, -1.5) * self.steps])
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
