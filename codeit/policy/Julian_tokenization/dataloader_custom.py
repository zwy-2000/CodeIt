import torch
from torch.utils.data import Dataset


class DataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the neural network for finetuning the model.
    """
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.input_ids = self.data['input_ids']
        self.labels = self.data['labels']
        self.task_ids = self.data['task_id']
        self.input_masks = self.data['attention_mask']
        self.input_max_length = 2048
        self.target_max_length = 512

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_id = self.input_ids[index]
        input_mask = self.input_masks[index]
        label_id = self.labels[index]

        input_id = torch.tensor(input_id + [0]*(self.input_max_length-len(input_id)))
        input_mask = torch.tensor(input_mask + [0]*(self.input_max_length-len(input_mask)))
        label_id = torch.tensor(label_id + [0]*(self.target_max_length-len(label_id)))
        label_mask = torch.tensor([1]*len(label_id) + [0]*(self.target_max_length-len(label_id)))

        # print(self.tokenizer.decode(input_id.tolist()))
        # print("input_id,", input_id.shape, input_id)
        # print("input_mask:", input_mask.shape, input_mask)
        # print("task_id:", task_id.shape, task_id)
        # print("task_mask:", task_mask.shape, task_mask)

        return {
            'source_ids': input_id,
            'source_mask': input_mask,
            'target_ids': label_id,
            'target_mask': label_mask,
            # 'name': name,
            # 'local_path': path,
            # 'percent_of_seen_pairs': percent_of_the_seen_pairs
        }
