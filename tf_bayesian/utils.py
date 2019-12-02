import numpy as np


class BatchManager:
    """
    Class to handle batching of numpy arrays for ML training loops
    Originally written by R. Solli and published under MIT
    """
    def __init__(self, nSamples, batch_size, shuffle=True):
        self.available = np.arange(0, nSamples)
        self.batch_size = batch_size
        self.remainder = nSamples % self.batch_size
        self.n = nSamples // self.batch_size
        # self.batch_multip = self.n*self.batch_size
        self.i = 0
        self.stopiter = False
        if shuffle:
            self.indices = np.random.choice(
                self.available, size=(self.n, self.batch_size), replace=False
            )
        else:
            ordinary = np.arange(0, batch_size * self.n)
            self.indices = ordinary.reshape((self.n, self.batch_size))

    def __next__(self,):

        if self.stopiter:
            raise StopIteration()
        elif self.i < self.n:
            # ind = np.random.choice(self.available, self.batch_size, replace=False)
            ind = self.indices[self.i]
            self.i += 1
        else:
            # ind = np.random.choice(self.available, self.nSamples, replace=False)
            if self.remainder == 0:
                raise StopIteration()
            ind = np.setdiff1d(self.available, self.indices.flatten())
            self.stopiter = True

        # self.available = np.delete(self.available, ind)
        # self.nSamples = self.available.shape[0]

        return ind
        # return self.X[ind].reshape(self.batch_size, self.n_pixels)

    def __iter__(self,):
        return self

    def __len__(self,):
        return self.n + 1


if __name__ == "__main__":
    import sys

    n_samp = 10
    batch_size = 3
    itercount = 0

    bm = BatchManager(n_samp, batch_size)

    for batch_ind in bm:

        print("-------------####-------------")

        itercount += 1
        if itercount > 10:
            sys.exit(0)

