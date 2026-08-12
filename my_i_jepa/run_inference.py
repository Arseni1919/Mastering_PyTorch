from config import config
from define_model import IJEPAModel
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from get_data import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collect_features_and_labels(n_per_class: int = 100):
    model = IJEPAModel(
        image_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        embed_dim=config.embed_size,
        encoder_depth=config.encoder_depth,
        encoder_num_heads=config.encoder_num_heads,
        predictor_embed_dim=config.predictor_embed_dim,
        predictor_depth=config.predictor_depth,
        predictor_num_heads=config.predictor_num_heads,
        sigreg_num_slices=config.sigreg_num_slices,
        sigreg_lambda=config.sigreg_lambda,
        M=4,
        target_scale_range=config.target_scale_range,
        target_aspect_ratio_range=config.target_aspect_ratio_range,
        context_scale_range=config.context_scale_range
    )
    model.load_state_dict(torch.load('state_dict.pt', map_location=device))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Group images by label
    data_dict = {}
    for x, y in dataset:
        key = int(y)
        data_dict.setdefault(key, []).append(x)

    features = []
    labels = []

    with torch.no_grad():
        for label, imgs in data_dict.items():
            for x in imgs[:n_per_class]:
                x = x.to(device)
                if x.dim() == 3:          # (C, H, W) -> add batch dim
                    x = x.unsqueeze(0)

                pred = model.encoder(x)   # likely (1, num_patches, embed_dim)

                # Pool patch tokens down to a single vector per image.
                # If your encoder already returns (1, embed_dim) or (embed_dim,),
                # this is a no-op / safe fallback.
                if pred.dim() == 3:
                    pred = pred.mean(dim=1)   # (1, embed_dim)
                pred = pred.squeeze(0)        # (embed_dim,)

                features.append(pred.cpu().numpy())
                labels.append(label)

    X = np.stack(features)          # (N, embed_dim)
    y = np.array(labels)            # (N,)
    return X, y


def plot_tsne(X, y, save_path: str = 'tsne_plot.png', use_pca_init: bool = True):
    # For high-dim embeddings, PCA first speeds up t-SNE and denoises a bit.
    if use_pca_init and X.shape[1] > 50:
        X = PCA(n_components=50, random_state=42).fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(X) // 100)),
        learning_rate='auto',
        init='pca',
        random_state=42,
    )
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(9, 9))
    classes = np.unique(y)
    cmap = plt.get_cmap('tab10', len(classes))

    for i, c in enumerate(classes):
        mask = y == c
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            s=8, alpha=0.7, color=cmap(i), label=str(c)
        )

    plt.legend(title='digit', markerscale=2, loc='best')
    plt.title('t-SNE of IJEPA encoder features (MNIST)')
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    X, y = collect_features_and_labels(n_per_class=100)
    plot_tsne(X, y)


if __name__ == '__main__':
    main()