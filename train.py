from ultralytics import YOLO
import os
import sys
import argparse

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLO model for surgical tool detection')
    parser.add_argument('model', type=str, nargs='?', default='m', 
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size: n (nano), s (small), m (medium), l (large), x (xlarge). Default: m')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size. Use -1 for auto-batch. Default: auto based on model size')
    parser.add_argument('--cache', type=str, default=None, choices=['ram', 'disk', 'false'],
                        help='Cache mode: ram, disk, or false. Default: auto based on model size')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers. Default: 8')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs. Default: 300')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size. Default: 640')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (epochs). Default: 50')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device(s): single GPU (0), multiple GPUs (0,1,2,3), all GPUs (0,1,2,3,...), or cpu. Default: 0')
    
    args = parser.parse_args()
    model_size = args.model
    
    # Validate and process device parameter
    device = args.device
    if device.lower() != 'cpu':
        # Validate GPU device format (should be comma-separated integers)
        try:
            gpu_ids = [int(x.strip()) for x in device.split(',')]
            device = ','.join(map(str, gpu_ids))  # Normalize format
        except ValueError:
            print(f"Error: Invalid device format '{args.device}'")
            print("Device should be: '0' (single GPU), '0,1,2' (multiple GPUs), or 'cpu'")
            sys.exit(1)
    
    # Configure paths and settings based on model size
    model_name = f'yolo12{model_size}'
    checkpoint_path = f'runs/detect/{model_name}_surgical_tool_detection/weights/last.pt'
    base_model_path = f'base_models/{model_name}.pt'
    
    # Model-specific default configurations
    default_config = {
        'n': {'batch': -1, 'cache': 'ram', 'save_period': 10},
        's': {'batch': -1, 'cache': 'ram', 'save_period': 10},
        'm': {'batch': -1, 'cache': 'ram', 'save_period': 10},
        'l': {'batch': -1, 'cache': 'ram', 'save_period': 5},
        'x': {'batch': -1, 'cache': 'disk', 'save_period': 5}
    }
    
    # Override defaults with command-line arguments
    batch_size = args.batch if args.batch is not None else default_config[model_size]['batch']
    cache_mode = args.cache if args.cache is not None else default_config[model_size]['cache']
    if cache_mode == 'false':
        cache_mode = False
    
    config = {
        'batch': batch_size,
        'cache': cache_mode,
        'workers': args.workers,
        'save_period': default_config[model_size]['save_period'],
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'patience': args.patience,
        'device': device
    }
    
    print(f"========================================")
    print(f"Training YOLOv12{model_size.upper()} Model")
    print(f"========================================")
    print(f"Base model: {base_model_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Configuration:")
    print(f"  - Batch size: {config['batch']}")
    print(f"  - Cache mode: {config['cache']}")
    print(f"  - Workers: {config['workers']}")
    print(f"  - Epochs: {config['epochs']}")
    print(f"  - Image size: {config['imgsz']}")
    print(f"  - Patience: {config['patience']}")
    print(f"  - Device: {config['device']}")
    print(f"========================================\n")
    
    # Check if a checkpoint exists to resume training
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        # When resuming, just call train() with resume=True
        results = model.train(resume=True)
    else:
        print(f"Starting new training from base model: {base_model_path}")
        
        # Check if base model exists
        if not os.path.exists(base_model_path):
            print(f"Error: Base model not found at {base_model_path}")
            print(f"Please download it first or run setup script.")
            sys.exit(1)
        
        # Load a pre-trained model
        model = YOLO(base_model_path)

        # Train the model with configuration
        results = model.train(
            data='dataset/data.yaml',
            device=config['device'],
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            name=f'{model_name}_surgical_tool_detection',
            patience=config['patience'],
            save_period=config['save_period'],
            cache=config['cache'],
            workers=config['workers'],
            amp=True  # Automatic Mixed Precision for faster training
        )

    print(f"\n========================================")
    print(f"Training Complete!")
    print(f"========================================\n")
    
    # Evaluate the model's performance on the validation set
    print("Evaluating model on validation set...")
    results = model.val()

    # Export the model to ONNX format
    print("\nExporting model to ONNX format...")
    success = model.export(format='onnx')
    
    if success:
        print(f"✓ Model exported successfully!")
        print(f"ONNX model saved to: runs/detect/{model_name}_surgical_tool_detection/weights/best.onnx")