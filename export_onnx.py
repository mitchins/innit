import torch
import onnx
from train_innit import TinyByteCNN_EN, PAD
import os

def export_to_onnx():
    """Export trained PyTorch model to ONNX format"""
    print("Loading trained model...")
    
    # Load trained model
    model = TinyByteCNN_EN().eval()
    
    try:
        model.load_state_dict(torch.load("artifacts/innit.pt", map_location="cpu"))
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first with: python train_innit.py")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Create dummy input for export
    dummy_input = torch.full((1, 2048), PAD, dtype=torch.long)
    
    print("Exporting to ONNX...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            "artifacts/innit.onnx",
            opset_version=17,
            input_names=["tokens"],
            output_names=["logits"],
            dynamic_axes={
                "tokens": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            export_params=True,
            do_constant_folding=True,
            verbose=False
        )
        print("ONNX export successful: artifacts/innit.onnx")
        
        # Verify the exported model
        onnx_model = onnx.load("artifacts/innit.onnx")
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")
        
        # Print model size info
        model_size = os.path.getsize("artifacts/innit.onnx") / (1024 * 1024)
        print(f"Model size: {model_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        return False

def quantize_model():
    """Optional: Quantize the ONNX model to int8 for smaller size"""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        print("Quantizing model to int8...")
        quantize_dynamic(
            "artifacts/innit.onnx",
            "artifacts/innit_int8.onnx",
            weight_type=QuantType.QInt8
        )
        
        # Compare sizes
        original_size = os.path.getsize("artifacts/innit.onnx") / (1024 * 1024)
        quantized_size = os.path.getsize("artifacts/innit_int8.onnx") / (1024 * 1024)
        
        print(f"Original model: {original_size:.2f} MB")
        print(f"Quantized model: {quantized_size:.2f} MB")
        print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
    except ImportError:
        print("Quantization skipped - onnxruntime quantization not available")
    except Exception as e:
        print(f"Quantization failed: {e}")

if __name__ == "__main__":
    if export_to_onnx():
        quantize_model()
    else:
        print("Export failed")