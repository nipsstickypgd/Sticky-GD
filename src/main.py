from typing import List

import numpy as np
import torch

import utils
from datasets.TextDataset import TextDataset
from optimizers.Optimizer import Optimizer
from optimizers.ao_admm import AO_ADMM
from optimizers.block_coordinate import BlockCoordinate
from optimizers.least_squares import LeastSquares
from optimizers.spgd import SPGD
from optimizers.mult_update import MultUpdate
import time


def main():
    runs = 10
    # ds = RandomDataset()
    # ds = ImageDataset()
    ds = TextDataset()
    # ds = RecommendationDataset()
    k = ds.k()

    for run in range(runs):
        step = ds.step(run)
        # M = torch.from_numpy(ds.generate(run)).float().to(device)
        M, E = ds.generate(run)
        M = utils.to_tensor(M)
        E = utils.to_tensor(E)
        # M = ds.generate(run)

        print(M.shape)
        names = []
        objs = []
        times = []

        algorithms: List[Optimizer] = [
            SPGD(M, k, step, True),
            MultUpdate(M, k),
            BlockCoordinate(M, k, step, False),
            AO_ADMM(M, k),
            LeastSquares(M, k)
        ]
        for opt in algorithms:
            print(opt.name())
            names.append(opt.name())
            start_time = time.time()
            W, H = opt.optimize(E, ds.iterations())
            print()
            objs.append(utils.objective(M, W, H, E))
            elapsed = time.time() - start_time
            times.append(elapsed)
            print("time", elapsed)
            ds.postprocess(W, H, run, opt.short_name())

    print(names)
    print(objs)
    print(times)


if __name__ == '__main__':
    main()
