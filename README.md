# FaceUV: FLAME UV Coordinate Estimation Network

FaceUV is a deep learning model for estimating FLAME UV coordinates per pixel from face images. Built on DINOv3 backbone, it predicts dense UV mappings for facial geometry.

## Overview

This project estimates UV coordinates of the FLAME (Faces Learned with an Articulated Model and Expressions) model for each pixel in a face image. The UV coordinates allow mapping 2D image pixels to 3D FLAME mesh surface locations.

## Architecture

- **Encoder**: VGG19 + DINOv3 (frozen) for multi-scale feature extraction
- **Decoder**: Coarse-to-fine architecture with feature fusion modules
- **Output**: 3 channels (U coordinate, V coordinate, mask)

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
FaceUV/
├── configs/
│   └── default_config.yaml     # Training configuration
├── src/
│   ├── models/
│   │   ├── face_uv_model.py    # Main model definition
│   │   ├── encoder.py          # VGG19 + DINOv3 encoder
│   │   ├── decoder.py          # Coarse-to-fine decoder
│   │   └── dinov3/             # DINOv3 implementation
│   ├── data/
│   │   └── face_uv_datamodule.py  # Data loading
│   └── utils/
│       ├── visualization.py    # Visualization utilities
│       └── postprocess.py      # Postprocessing utilities
├── checkpoints/                # Model checkpoints
├── experiments/                # Training logs and outputs
├── train.py                    # Training script
└── inference.py                # Inference script
```

## Dataset Format

Expected data structure:

```
data_dir/
├── images/          # Input RGB images
├── uvs/             # Ground truth UV maps (16-bit PNG, 2 channels)
├── masks/           # Face masks
└── calibs/          # Camera calibration (optional)
```

UV maps should be stored as 16-bit PNG images where:
- Red channel: U coordinate (0-65535 maps to 0-1)
- Green channel: V coordinate (0-65535 maps to 0-1)

## Training

1. Prepare your dataset following the format above
2. Update `configs/default_config.yaml` with your dataset paths
3. Run training:

```bash
python train.py --config configs/default_config.yaml
```

## Inference

Run inference on images:

```bash
python inference.py \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --checkpoint experiments/checkpoints/decoder.ckpt \
    --dinov3_checkpoint checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

## Output

The model outputs:
- **UV Map**: 2-channel map with U and V coordinates (range: [0, 1])
- **Mask**: Binary mask indicating valid face regions
- **Visualization**: RGB visualization where R=U, G=V, B=Mask

## Key Differences from FaceDepth

- **Output**: UV coordinates (2D surface coords) instead of depth (1D distance)
- **Loss**: L1 + L2 loss on UV coordinates instead of depth values
- **Normalization**: UV coords normalized to [-1, 1] during training, [0, 1] at inference
- **Use Case**: 3D texture mapping and correspondence instead of depth estimation

## Model Details

### Input
- RGB image (448x448)
- Normalized with ImageNet statistics

### Output
- UV map: 2 channels, range [0, 1]
- Mask: 1 channel, range [0, 1]

### Training
- Optimizer: AdamW
- Learning rate: 1e-3
- Scheduler: CosineAnnealingLR
- Loss: L1 + 0.5*L2 (UV) + 0.01*BCE (mask)
- Auxiliary losses for deep supervision

## Citation

If you use this code, please cite the relevant papers for DINOv3 and FLAME.

## License

See LICENSE file for details.

## Acknowledgments

- Based on FaceDepth architecture
- DINOv3 backbone from Meta AI
- FLAME model for face representation
