from openfl.utilities.data_splitters import DirichletNumPyDataSplitter
from openfl.utilities.data_splitters import EqualNumPyDataSplitter
from openfl.utilities.data_splitters import RandomNumPyDataSplitter
from openfl.utilities.data_splitters import NumPyDataSplitter

from tqdm import trange

import numpy as np

def get_label_count(labels, label):
    """Count samples with label `label` in `labels` array."""
    return len(np.nonzero(labels == label).ravel())


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
        return idx_data

class LogNormalNumPyDataSplitter(NumPyDataSplitter):
    """Unbalanced (LogNormal) dataset split.

    This split assumes only several classes are assigned to each collaborator.
    Firstly, it assigns classes_per_col * min_samples_per_class items of dataset
    to each collaborator so all of collaborators will have some data after the split.
    Then, it generates positive integer numbers by log-normal (power) law.
    These numbers correspond to numbers of dataset items picked each time from dataset
    and assigned to a collaborator.
    Generation is repeated for each class assigned to a collaborator.
    This is a parametrized version of non-i.i.d. data split in FedProx algorithm.
    Origin source: https://github.com/litian96/FedProx/blob/master/data/mnist/generate_niid.py#L30

    NOTE: This split always drops out some part of the dataset!
    Non-deterministic behavior selects only random subpart of class items.
    """

    def __init__(self, mu,
                 sigma,
                 num_classes,
                 classes_per_col,
                 min_samples_per_class,
                 seed=0):
        """Initialize the generator.

        Args:
            mu(float): Distribution hyperparameter.
            sigma(float): Distribution hyperparameter.
            classes_per_col(int): Number of classes assigned to each collaborator.
            min_samples_per_class(int): Minimum number of collaborator samples of each class.
            seed(int): Random numbers generator seed.
                For different splits on envoys, try setting different values for this parameter
                on each shard descriptor.
        """
        self.mu = mu
        self.sigma = sigma
        self.num_classes = num_classes
        self.classes_per_col = classes_per_col
        self.min_samples_per_class = min_samples_per_class
        self.seed = seed

    def split(self, data, num_collaborators):
        """Split the data.

        Args:
            data(np.ndarray): numpy-like label array.
            num_collaborators(int): number of collaborators to split data across.
                Should be divisible by number of classes in ``data``.
        """
        np.random.seed(self.seed)
        print("Ola Amigos")
        idx = [[] for _ in range(num_collaborators)]

        data = np.argmax(data, axis=1)
        samples_per_col = self.classes_per_col * self.min_samples_per_class
        for col in range(num_collaborators):
            for c in range(self.classes_per_col):
                label = (col + c) % self.num_classes
                label_idx = np.nonzero(data == label).ravel()
                slice_start = col // self.num_classes * samples_per_col
                slice_start += self.min_samples_per_class * c
                slice_end = slice_start + self.min_samples_per_class
                print(f'Assigning {slice_start}:{slice_end} of class {label} to {col} col...')
                idx[col] += label_idx[slice_start:slice_end].tolist()
#         if any([len(i) != samples_per_col for i in idx]):
#             raise SystemError(f'''All collaborators should have {samples_per_col} elements
# but distribution is {[len(i) for i in idx]}''')

        props_shape = (
            self.num_classes,
            num_collaborators // self.num_classes,
            self.classes_per_col
        )
        props = np.random.lognormal(self.mu, self.sigma, props_shape)
        num_samples_per_class = [[[get_label_count(data, label) - self.min_samples_per_class]]
                                 for label in range(self.num_classes)]

        num_samples_per_class = np.array(num_samples_per_class)
        props = num_samples_per_class * props / np.sum(props, (1, 2), keepdims=True)
        for col in trange(num_collaborators):
            for j in range(self.classes_per_col):
                label = (col + j) % self.num_classes
                num_samples = int(props[label, col // self.num_classes, j])

                print(f'Trying to append {num_samples} samples of {label} class to {col} col...')
                slice_start = np.count_nonzero(data[np.hstack(idx)] == label)
                slice_end = slice_start + num_samples
                label_count = get_label_count(data, label)
                if slice_end < label_count:
                    label_subset = np.nonzero(data == (col + j) % self.num_classes)[0]
                    idx_to_append = label_subset[slice_start:slice_end]
                    idx[col] = np.append(idx[col], idx_to_append)
                else:
                    print(f'Index {slice_end} is out of bounds '
                          f'of array of length {label_count}. Skipping...')
        print(f'Split result: {[len(i) for i in idx]}.')
        print([np.unique(data[i]) for i in idx])
        return idx




class OneClassSplitter(NumPyDataSplitter):

    def __init__(self):
        self.n_class = 5
        n_collaborators = 5

    def split(self, data, num_collaborator):
        idx = []
        data = np.argmax(data, axis=1)
        for label in range(self.n_class):
            label_idx = np.nonzero(data == label).ravel()
            idx.append(label_idx)

        print("asas  sdjojskljklsjkljlsjd")
        print([len(i) for i in idx])
        return idx


class FourClassSplitter(NumPyDataSplitter):

    def __init__(self, n_classes_per_collab, n_classes, num_collaborator):
        self.name = "4 Class Splitter"
        self.n_classes_per_collab = n_classes_per_collab
        self.n_classes = n_classes
        self.num_collaborator = num_collaborator

    def split(self, data, num_collaborator):
        label = 0
        colab_to_class = [[] for i in range(self.num_collaborator)]
        class_to_colab = [[] for i in range(self.n_classes)]

        data = np.argmax(data, axis=1)

        for colab_id, colab_list in enumerate(colab_to_class):
            for i in range(self.n_classes_per_collab):
                colab_list.append(label)
                class_to_colab[label].append(colab_id)
                label = (label + 1) % self.n_classes

        final_idx = [[] for i in range(self.num_collaborator)]
        for label in range(self.n_classes):
            label_idx = np.nonzero(data == label).ravel()
            ## Drops some indices
            tlen = int(len(label_idx) / len(class_to_colab[label])) * len(class_to_colab[label])
            label_idx = label_idx[ : tlen]
            label_idx = label_idx.reshape(len(class_to_colab[label]), -1)

            for i, colab_id in enumerate(class_to_colab[label]):
                final_idx[colab_id].extend(label_idx[i, :].tolist())



        label_matrix = np.array([np.array([data[idx] for idx in idx_list]) for idx_list in final_idx])
        print([np.unique(label_mat) for label_mat in label_matrix])
        return final_idx










def SplitFunctionGenerator(splitter_name):
    if splitter_name == "Random-Equal-Split":
        return RandomNumPyDataSplitter()
    elif splitter_name == "Random-Unequal-Split":
        return EqualNumPyDataSplitter()
    elif splitter_name == "Equal-Equal-Split":
        return AllEqualDataSplitter()
    elif splitter_name == "2-Class-per-collab-Split":
        return FourClassSplitter(n_classes_per_collab = 2, n_classes = 5, num_collaborator = 5)
    elif splitter_name == "3-Class-per-collab-split":
        return FourClassSplitter(n_classes_per_collab = 3, n_classes = 5, num_collaborator = 5)
    elif splitter_name == "1-Class-per-collab-split":
        return FourClassSplitter(n_classes_per_collab = 1, n_classes = 5, num_collaborator = 5)
    elif splitter_name == "4-Class-per-collab-split":
        return FourClassSplitter(n_classes_per_collab = 4, n_classes = 5, num_collaborator = 5)


