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
        #Array for final indices of each collaborator
        idx_data = np.zeros((5,1))
        for i in range(num_classes):

            #Pick indices of data with class i
            idx_list = np.where(data[:, i] == 1)[0]
            #FInd the number of indices for each collaborator
            len_per_collaborator = (int(len(idx_list) / num_collaborators))
            #Find the total number of indices
            total_data_len = len_per_collaborator * num_collaborators
            #take the first total_data_len indices
            idx_list = idx_list[ : total_data_len]
            idx_list = idx_list.reshape(num_collaborators, len_per_collaborator)

            idx_data = np.append(idx_data, idx_list, axis=1)

        idx_data = np.delete(idx_data, 0, 1)
        print(idx_data)
        print(idx_data.shape)
        return idx_data










def SplitFunctionGenerator(splitter_name):
    if splitter_name == "Random-Equal-Split":
        return RandomNumPyDataSplitter()
    elif splitter_name == "Random-Unequal-Split":
        return EqualNumPyDataSplitter()
    elif splitter_name == "Equal-Equal-Split":
        return AllEqualDataSplitter()


