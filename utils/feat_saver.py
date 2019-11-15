

import numpy as np
import os
import pandas as pd

class Saver:

    def __init__(self):
        self.features = None
        self.labels = None


    def update (self, values, label=None):

        if self.features is None:
            self.features = values
            if label is not None:
                self.labels = label
        else:
            self.features = np.concatenate((self.features, values))
            if label is not None:
                self.labels = np.concatenate((self.labels, label))


    def save (self, path):

        if self.labels is not None:
            data = np.concatenate((self.labels, self.features), axis=1)
            cols = ['label', 'features']
        else:
            data = self.features
            cols = ['features']

        df = pd.DataFrame(data, columns=cols)
        df.to_csv(path, index=False)


