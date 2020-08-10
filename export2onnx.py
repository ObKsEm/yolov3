from models import *
import torch

weights = "weights/2020.07.15/best.pt"
cfg = "cfg/yolov3-spp"
imgsz = 608
f = "model.onnx"


def main():
    model = Darknet(cfg, imgsz)
    # device = torch_utils.select_device("0")
    model.load_state_dict(torch.load(weights)['model'])
    model.eval()
    img = torch.zeros((1, 3, imgsz, imgsz))  # (1, 3, 320, 192)
    torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                      input_names=['images'], output_names=['classes', 'boxes'])
    import onnx
    model = onnx.load(f)  # Load the ONNX model
    onnx.checker.check_model(model)  # Check that the IR is well formed
    print(onnx.helper.printable_graph(model.graph))


if __name__ == "__main__":
    main()
