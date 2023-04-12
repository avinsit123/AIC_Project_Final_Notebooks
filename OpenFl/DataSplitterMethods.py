from openfl.utilities.data_splitters import DirichletNumPyDataSplitter
from openfl.utilities.data_splitters import EqualNumPyDataSplitter
from openfl.utilities.data_splitters import LogNormalNumPyDataSplitter
from openfl.utilities.data_splitters import RandomNumPyDataSplitter
from openfl.utilities.data_splitters import NumPyDataSplitter


import numpy as np

class AllEqualDataSplitter(NumPyDataSplitter):

    """
    A Data splitter that splits equal number
    of equally amongst all the collaborators in the network
    """

    def __init__(self):
        self.name = "Splitting all equal data"


    def split(self, data, num_collaborators):

        ## With assumption we have 5 classes
        num_classes = 5
        idx_data = np.array(([[] for i in range(num_collaborators)]))
        for i in range(num_classes):

            idx_list = np.where(data[:, i] == 1)[0]
            len_per_collaborator = (int(len(idx_list) / num_collaborators))
            total_data_len = len_per_collaborator * num_collaborators
            idx_list = idx_list[ : total_data_len]
            idx_new_data = np.zeros((num_collaborators, idx_data.shape[1] + len_per_collaborator))
            if i != 0:
                idx_new_data[:, :idx_data.shape[1]] = idx_data

            for j in range(num_collaborators):
                idx_new_data[i, idx_data.shape[1]: ] = idx_list[j * len_per_collaborator: (j + 1) * len_per_collaborator]

            idx_data = idx_new_data

        for i in range(num_collaborators):
            print(len(idx_data[i]))
            np.random.shuffle(idx_data[i])

        return idx_data










def SplitFunctionGenerator(splitter_name):
    if splitter_name == "Random-Equal-Split":
        return RandomNumPyDataSplitter()
    elif splitter_name == "Random-Unequal-Split":
        return EqualNumPyDataSplitter()
    elif splitter_name == "Equal-Equal-Split":
        return AllEqualDataSplitter()


