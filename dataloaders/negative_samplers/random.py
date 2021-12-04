from .base import AbstractNegativeSampler

from tqdm import trange

import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        keys = range(1, self.item_count + 1)
        for user in trange(0, self.user_count):
            if isinstance(self.train[user][1], tuple):
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])

            samples = []
            # whole_candidates = [x for x in keys if x not in seen]
            # for _ in range(self.sample_size):
                # item = np.random.choice(self.item_count) + 1
                # while item in seen or item in samples:
                #     item = np.random.choice(self.item_count) + 1
                # samples.append(item)
            sample_id = np.random.choice(keys, self.sample_size + len(seen), replace=False)
            sample_ids = [x for x in sample_id if x not in seen]
            
            samples.extend(sample_ids[:])


            negative_samples[user] = samples[:self.sample_size]

        return negative_samples
