import torch
from get_data import dataset
from config import config
from define_model import DiTFullModel


def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
    example_x, _ = dataset[0]
    ch, height, width = example_x.shape
    net: DiTFullModel = DiTFullModel(
        side=config.side_size,
        path_side=config.patch_size,
        head_dim=config.hidden_size // config.num_heads,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes,
        class_emb_size=config.class_emb_size,
        time_emb_size=config.time_emb_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        channels=config.channels
    ).to(device)
    net.load_state_dict(state_dict=torch.load('saved_model.pt', map_location=device))
    net.eval()
    x = torch.randn((10, ch, height, width), device=device)
    y = torch.arange(0, 10, device=device).long()
    t = torch.rand((10,), device=device)
    torch.onnx.export(net, (x, t, y), f=config.onnx_path)


if __name__ == '__main__':
    main()
