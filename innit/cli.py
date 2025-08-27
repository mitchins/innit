"""CLI interface for innit - English vs non-English detection"""

import sys
import json
import argparse
from pathlib import Path

def find_model_path():
    """Find the innit ONNX model file"""
    # Check common locations
    candidates = [
        Path("artifacts/innit.onnx"),
        Path("innit.onnx"),
        Path("artifacts/innit_int8.onnx"),
        Path(__file__).parent.parent / "artifacts" / "innit.onnx",
        Path(__file__).parent / "innit.onnx",
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    return None

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Detect if text is English or not English",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  innit book.txt                    # Detect language of book.txt
  innit book.txt --model innit.onnx # Use specific model file
  innit book.txt --json            # Output as JSON
        """
    )
    
    parser.add_argument(
        "text_file", 
        help="Path to text file to analyze"
    )
    
    parser.add_argument(
        "--model", "-m",
        help="Path to ONNX model file (auto-detected if not provided)"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )
    
    parser.add_argument(
        "--window-size", "-w",
        type=int,
        default=2048,
        help="Window size for analysis (default: 2048)"
    )
    
    args = parser.parse_args()
    
    # Check if text file exists
    text_path = Path(args.text_file)
    if not text_path.exists():
        print(f"Error: File not found: {text_path}", file=sys.stderr)
        sys.exit(1)
    
    # Find model file
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}", file=sys.stderr)
            sys.exit(1)
    else:
        model_path = find_model_path()
        if model_path is None:
            print("Error: Could not find innit ONNX model file.", file=sys.stderr)
            print("Please specify model path with --model or ensure model is in artifacts/", file=sys.stderr)
            sys.exit(1)
    
    # Load text
    try:
        text = text_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Error reading text file: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not text.strip():
        result = {
            "label": "UNCERTAIN",
            "mean_pEN": 0.0,
            "hi>=0.99": 0.0,
            "windows": 0
        }
    else:
        # Run inference
        try:
            from .onnx_runner import ONNXInnitRunner, score_text_onnx
            
            runner = ONNXInnitRunner(model_path)
            result = score_text_onnx(runner, text, window_size=args.window_size)
            
        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Please install onnxruntime: pip install onnxruntime", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error during inference: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Output results
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"File: {text_path}")
        print(f"Language: {result['label']}")
        print(f"Mean P(English): {result['mean_pEN']:.4f}")
        print(f"High confidence windows (>=99%): {result['hi>=0.99']:.1%}")
        print(f"Number of windows: {result['windows']}")

if __name__ == "__main__":
    main()