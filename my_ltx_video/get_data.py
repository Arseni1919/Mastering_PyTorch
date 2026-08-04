import os
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import _T_co
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.v2.functional as F
import matplotlib.pyplot as plt


from config import config


class KTHActionDataset(Dataset):
    def __init__(self, root, num_frames, side_size):
        super().__init__()
        self.root = root
        self.num_frames = num_frames
        self.side_size = side_size
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(f'{root}/{d}')])
        self.samples = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root, class_name)
            for video_name in os.listdir(class_dir):
                path = os.path.join(class_dir, video_name)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> _T_co:
        path, label = self.samples[index]

        decoder = VideoDecoder(path)
        total_frames = len(decoder)
        start = random.randint(0, total_frames-self.num_frames)
        clip = decoder[start:start + self.num_frames]

        clip = F.center_crop(clip, (120, 120))
        clip = F.resize(clip, (self.side_size, self.side_size))
        clip = F.rgb_to_grayscale(clip, num_output_channels=1)

        clip = clip.permute(1,0,2,3)
        clip = clip.float() / 255.0

        return clip, label

root = os.environ.get('KTH_DATA_ROOT', 'KTH_Action')
dataset = KTHActionDataset(
    root=root,
    num_frames=config.num_frames,
    side_size=config.side_size
)


def main():
    for v in range(10):
        x, y = dataset[random.randint(0, len(dataset)-1)]
        for i in range(config.num_frames):
            plt.cla()
            frame = x[0,i,:,:].unsqueeze(2)
            plt.imshow(frame)
            plt.title(f'[{v}] frame {i}')
            plt.pause(0.01)
    plt.show()


if __name__ == '__main__':
    main()
