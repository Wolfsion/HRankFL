# data.samplers.py
"""
@deprecated(version='0.1', reason="This class or function is not perfect")
class LabelParser():
    def __init__(self, datatype):
        self.indices = []
        self.labels = []
        self.labelToIndices(datatype)

    def labelToIndices(datatype):
        if datatype == DataSetType.CIFAR10.value:
            pass
        else:
            total_num = 0

@deprecated(version='0.1', reason="This class or function is not perfect")
class NIIDSampler(LSampler):
    def getIndices(self, datatype, num_slices, num_round, data_per_client, dataset,
                   client_selection, client_per_round=None):
        if datatype == DataSetType.CIFAR10.value:
            total_num = CIFAR10_NUM_TRAIN_DATA
            total_class = CIFAR10_CLASSES
        else:
            total_num = 0

        lapr = LabelParser(datatype)
        len_slice = total_num // num_slices

        # depend on label
        tmp_indices = []

        range_partition = list(range(num_slices))
        new_list_ind = [[] for _ in range(num_slices)]

        if client_selection:
            assert client_per_round is not None
            assert client_per_round <= num_slices

        list_pos = [0] * num_slices
        for _ in range(num_round):
            if client_selection:
                selected_client_idx = random.sample(range_partition, client_per_round)
            else:
                selected_client_idx = range_partition

        for client_idx in selected_client_idx:
            ind = tmp_indices[client_idx]
            pos = list_pos[client_idx]
            while len(new_list_ind[client_idx]) < pos + data_per_client:
                random.shuffle(ind)
                new_list_ind[client_idx].extend(ind)
            self.indices.extend(new_list_ind[client_idx][pos:pos + data_per_client])
            list_pos[client_idx] = pos + data_per_client
"""
