import numpy as np
import torch


class Dataloader:
    def __init__(self, ru_path, en_path):

        self.data = {"ru": [], "en": []}

        with open(ru_path, "r") as ru, open(en_path, "r") as en:
            for ru_line, en_line in zip(ru, en):
                _ru_line = ru_line.split(" ")
                _en_line = en_line.split(" ")

                if (
                    len(_ru_line) > 2
                    and len(_en_line) > 2
                    and len(_ru_line) < 150
                    and (len(_en_line) < 150)
                ):
                    self.data["ru"].append([int(idx) for idx in _ru_line])
                    self.data["en"].append([int(idx) for idx in _en_line])

    def next_batch(self, batch_size, device):
        indices = np.random.randint(len(self.data["ru"]), size=batch_size)

        ru_input = [self.data["ru"][i] for i in indices]
        en_input = [self.data["en"][i] for i in indices]

        return self.torch_batch(en_input, ru_input, device)

    def iterate(self, device):
        for en_input, ru_input in zip(self.data["en"], self.data["ru"]):
            yield self.torch_batch([en_input], [ru_input], device)

    @staticmethod
    def torch_batch(input, target, device):
        input = Dataloader.padd_sequences(input)
        decoder_input = Dataloader.padd_sequences([line[:-1] for line in target])
        target = Dataloader.padd_sequences([line[1:] for line in target])

        return tuple(
            [
                torch.tensor(val, dtype=torch.long).to(device, non_blocking=True)
                for val in [input, decoder_input, target]
            ]
        )

    @staticmethod
    def padd_sequences(lines):
        lengths = [len(line) for line in lines]
        max_length = max(lengths)

        return np.array(
            [line + [0] * (max_length - lengths[i]) for i, line in enumerate(lines)]
        )
