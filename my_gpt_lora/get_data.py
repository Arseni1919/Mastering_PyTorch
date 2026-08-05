from torch.utils.data.dataset import _T_co
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
from config import config

class ShakespeareDataset(Dataset):
    def __init__(self, teken_ids, block_size):
        super().__init__()
        self.token_ids = teken_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids.data['input_ids']) - self.block_size

    def __getitem__(self, index) -> _T_co:
        return torch.tensor(self.token_ids.data['input_ids'][index:index+self.block_size])


tokenizer = AutoTokenizer.from_pretrained('gpt2')
with open('tinyshakespeare.txt', 'r') as r:
    data = r.read()
token_ids = tokenizer(data)
dataset = ShakespeareDataset(token_ids, block_size=config.block_size)


def main():
    batch = torch.stack([dataset[i] for i in range(8)])
    print(batch.shape)


if __name__ == '__main__':
    main()