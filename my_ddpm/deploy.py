import torch
from get_data import dataset
from config import config
from define_model import MyUNetClassConditionedModel


def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
    example_x, _ = dataset[0]
    ch, height, width = example_x.shape
    net = MyUNetClassConditionedModel(example_x=example_x).to(device)
    net.load_state_dict(state_dict=torch.load('saved_model.pt', map_location=device))
    net.eval()
    x = torch.randn((10, ch, height, width), device=device)
    y = torch.arange(0, 10, device=device).long()
    adapted_timestamp = torch.tensor(999).expand(x.shape[0]).to(device)
    torch.onnx.export(net, (x, adapted_timestamp, y), f=config.onnx_path)


if __name__ == '__main__':
    main()
