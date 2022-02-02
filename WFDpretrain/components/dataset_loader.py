from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import torch


class DatasetLoader_TracesAccu(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args, source_data, meta_split, train_aug=False):

        self.source_data = source_data
        self.meta_split = meta_split
        data_index = []
        labeles = []
        dict_label_indexes = self.meta_split[setname]
        for idx, label in enumerate(dict_label_indexes):
            for _, trace_index in enumerate(dict_label_indexes[label]):
                labeles.append(idx)
                data_index.append(trace_index)
        self.data_index = data_index
        self.label = labeles
        #random 
        per = np.random.permutation(len(self.data_index))	
        self.label = np.array(self.label)
        self.label =  self.label[per]
        self.data_index = np.array(self.data_index)
        self.data_index = self.data_index[per]		
        self.num_class = len(set(labeles))

        # Transformation
        if train_aug:
            # trace_length = 5000
            self.transform = transforms.Compose([
                PaddingWithoutLabel(512),
                RotateWithoutLabel(20), 
                MaskWithoutLabel(20), 
                ToTensor()])
        else:
            # trace_length = 5000
            self.transform = transforms.Compose([
                PaddingWithoutLabel(512),
                ToTensor()])


    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, i):
        index, label = self.data_index[i], self.label[i]
        trace = self.source_data[index]
        a = trace.astype(np.int32)
        trace = self.transform(a)
        return trace, label



class ToTensor(object):
    """
    Args:
        add one channel at first dim
    """

    def __init__(self):
        pass

    def __call__(self, trace):
        """
        Args:
            trace (numpy array): an sequence like "[+1, -1, +1, -1,...]"
        Returns:
            trace whose length is max_length, truncated if exceeded or padded if less(numpy array): an sequence like "[+1, -1, +1, -1,...]": PIL image.
        """
        trace = np.expand_dims(trace, axis=0)
        trace = trace.astype(np.float32)
        trace = torch.from_numpy(trace)

        return trace

class PaddingWithoutLabel(object):
    """
    Args:
        max_length （int）: the length of all traces should be this value.
    """

    def __init__(self, max_length):
        assert isinstance(max_length, int)
        self.max_length = max_length

    def __call__(self, trace):
        """
        Args:
            trace (numpy array): an sequence like "[+1, -1, +1, -1,...]"
        Returns:
            trace whose length is max_length, truncated if exceeded or padded if less(numpy array): an sequence like "[+1, -1, +1, -1,...]": PIL image.
        """
        if len(trace) < self.max_length:
            padded_trace = np.pad(trace, (0, self.max_length - len(trace)), constant_values=(0,0))
            return padded_trace
        else:
            return trace[:self.max_length]



class RotateWithoutLabel(object):
    """rotate sequence randomly, forward or backward
    Args:
        max_steps （int）: the step of rotation should be sampled from interval of 0 and max_steps.
        if max_steps is greater than 0, the interval is [0, max_steps] and rotate forward.
        if max_steps is less than 0, the interval is [max_steps, 0] and rotate backward.
    """

    def __init__(self, max_steps):
        assert isinstance(max_steps, int)
        self.max_steps = max_steps

    def __call__(self, trace):
        """
        Args:
            trace (numpy array): an sequence like "[+1, -1, +1, -1,...]"
        Returns:
            the rotated trace.
        """
        rotated_trace = np.roll(trace, int(random.random() * self.max_steps), axis= 0)
        return rotated_trace


class MaskWithoutLabel(object):
    """erase a subsequence randomly
    Args:
        mask_len （int）: the steps of subtraces would be masked from the original trace.
        But the start poing of the subtraces would be randomed.
    """

    def __init__(self, mask_len):
        assert isinstance(mask_len, int)
        self.mask_len = mask_len

    def __call__(self, trace):
        """
        Args:
            trace (numpy array): an sequence like "[+1, -1, +1, -1,...]"
        Returns:
            the trace whose subtrace(start point is random selected) have been masked.
        """
        trace_len = len(trace)
        missing_start_point = np.random.randint(0, trace_len)
        zero_nparr = np.array( [0]*(min(int(missing_start_point + self.mask_len) ,trace_len) - missing_start_point) )
        zero_nparr = np.reshape(zero_nparr.shape[0],1)
        trace[missing_start_point: min(int(missing_start_point + self.mask_len) ,trace_len)] = zero_nparr
        return trace
