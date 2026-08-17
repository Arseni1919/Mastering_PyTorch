import matplotlib.pyplot as plt
import torchvision
import random
import torch
import os

from config import config

# dataset = torchvision.datasets.MNIST(
#     root='mnist',
#     train=True,
#     download=True,
#     transform=torchvision.transforms.ToTensor()
# )

root = os.environ.get('STL10_DATA_ROOT', 'stl10')

dataset = torchvision.datasets.STL10(
    root=root,
    split="train",
    download=True,
    transform= torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(config.image_size),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

def sample_block_mask(num_patches_per_side, scale_range, aspect_ratio_range):
    scale_range_bottom, scale_range_up = scale_range
    scale_ratio = random.random() * (scale_range_up - scale_range_bottom) + scale_range_bottom
    target_area = (num_patches_per_side ** 2) * scale_ratio
    aspect_ratio_range_bottom, aspect_ratio_range_up = aspect_ratio_range
    aspect_ratio = random.random() * (aspect_ratio_range_up - aspect_ratio_range_bottom) + aspect_ratio_range_bottom
    height = min(int((target_area / aspect_ratio) ** 0.5), num_patches_per_side)
    width = min(int((target_area * aspect_ratio) ** 0.5), num_patches_per_side)
    corner_h = random.randint(0, num_patches_per_side - height)
    corner_w = random.randint(0, num_patches_per_side - width)
    h_range = torch.arange(corner_h, corner_h + height).reshape((height, 1))
    w_range = torch.arange(corner_w, corner_w + width).unsqueeze(0)
    final_set_of_patch_indices_t = (h_range * num_patches_per_side + w_range).flatten()
    return final_set_of_patch_indices_t


def sample_context_and_targets(
        num_patches_per_side, M, target_scale_range, target_aspect_ratio_range, context_scale_range
):
    target_masks = [
        sample_block_mask(num_patches_per_side, target_scale_range, target_aspect_ratio_range) for _ in range(M)
    ]
    context_mask = sample_block_mask(num_patches_per_side, context_scale_range, aspect_ratio_range=(1.0, 1.0))
    for target_mask in target_masks:
        context_mask = context_mask[torch.isin(context_mask, target_mask, invert=True)]
    return context_mask, target_masks



def main():
    num_patches_per_side = int(config.num_patches ** 0.5)
    context_mask, target_masks = sample_context_and_targets(
        num_patches_per_side, 4,
        config.target_scale_range, config.target_aspect_ratio_range, config.context_scale_range
    )
    print(f'{num_patches_per_side=}')
    print(f'{context_mask.shape=}')
    for target_mask in target_masks:
        print(f'- {target_mask.shape=}')

    x, y = dataset[random.randint(0, len(dataset) - 1)]
    Ch, H, W = x.shape
    print(f'{x.shape=}')  # C, S, S
    patches = x.reshape((Ch, num_patches_per_side, config.patch_size, num_patches_per_side, config.patch_size))
    patches = patches.transpose(2, 3).reshape((Ch, config.num_patches, config.patch_size ** 2))
    context_pic = patches.clone().detach()
    context_filter = torch.zeros_like(context_pic)
    context_filter[:,context_mask,:] = 1
    context_pic = context_pic * context_filter
    context_pic = context_pic.reshape(((Ch, num_patches_per_side, num_patches_per_side, config.patch_size, config.patch_size)))
    context_pic = context_pic.transpose(2, 3).reshape((Ch, H, W)).permute(1, 2, 0).numpy()
    target_pics = []
    for target_mask in target_masks:
        target_pic = patches.clone().detach()
        target_pic[:, target_mask, :] = 0
        target_pic = target_pic.reshape(
            ((Ch, num_patches_per_side, num_patches_per_side, config.patch_size, config.patch_size))
        )
        target_pic = target_pic.transpose(2, 3).reshape((Ch, H, W)).permute(1, 2, 0)
        target_pics.append(target_pic.numpy())


    fig, ax = plt.subplots(1, 6)
    ax[0].imshow(x.permute(1, 2, 0).numpy())
    ax[1].imshow(context_pic)
    for i, target_pic in enumerate(target_pics):
        ax[2 + i].imshow(target_pic)
    plt.show()



if __name__ == '__main__':
    main()