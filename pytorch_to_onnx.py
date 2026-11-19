import torch.onnx
import onnx
from pathlib import Path
from stanford_dogs.model import build_model
from stanford_dogs.config import NUM_CLASSES, IMG_SIZE, Final_PT_Model_Path, ONNX_Model_Path

def export_to_onnx():

    model = build_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(Final_PT_Model_Path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1,3,IMG_SIZE,IMG_SIZE, requires_grad=False)
    output_path = ONNX_Model_Path

    ONNX_Model_Path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        external_data=False,
        input_names=["image"],
        output_names=["dog_breed"],
        dynamic_axes={  # allow variable batch size at inference
            "image": {0: "batch_size"},
            "dog_breed": {0: "batch_size"},
        },
    )

    # validate the export
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)  
    print(f"[SUCCESS] ONNX model exported and verified successfully! ({output_path.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == '__main__':
    export_to_onnx()
