import torch


class Beam:
    def __init__(self, prob, data, cache, ended=False):
        self.prob = prob
        self.data = data
        self.cache = cache
        self.ended = ended

    @staticmethod
    def start_search(probs, n_beams, cache):
        probs, idx = torch.topk(probs, n_beams)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()

        return [
            Beam(prob=p, data=[id], cache=[val for val in cache])
            for id, p in zip(idx, probs)
        ]

    @staticmethod
    def update(beams, probs):
        n_beams = len(beams)
        v_s = probs.size(1)

        for i in range(n_beams):
            probs[i] *= float(beams[i].prob)

        probs = probs.view(-1)
        _, indices = torch.topk(probs, n_beams)
        indices = indices.cpu().numpy()

        indices = [(i % v_s, i // v_s) for i in indices]

        _probs = [[beams[j], i, probs[j * v_s + i]] for i, j in indices]

        return [
            Beam(p, b.data + [i], [val for val in b.cache], i == 2)
            for b, i, p in _probs
        ]
