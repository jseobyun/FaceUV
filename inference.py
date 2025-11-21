import os
import cv2
import argparse
import torch
import open3d as o3d
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from typing import Optional

from src.models import FaceUVModel
from src.utils.visualization import FaceDepthVisualizer  # Can be adapted for UV visualization
from src.utils.postprocess import FaceDepthConverter  # Can be adapted for UV postprocessing


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Face UV Coordinate Estimation Inference')

    parser.add_argument('--input_dir', type=str, default="/home/jseob/Downloads/TEST/dxf_test/images",
                        help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='/home/jseob/Downloads/TEST/dxf_test/uv_output',
                        help='Path to output directory')
    parser.add_argument('--checkpoint', type=str, default="experiments/checkpoints/face_uv/decoder.ckpt",
                        help='Path to decoder checkpoint (without dinov3)')
    parser.add_argument('--dinov3_checkpoint', type=str, default="checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                        help='Path to dinov3 checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    parser.add_argument('--image_size', type=int, nargs=2, default=[448, 448],
                        help='Input image size (height, width)')
    parser.add_argument('--output_channels', type=int, default=3,
                        help='Number of output channels (U + V + mask)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing multiple images')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save results')

    return parser.parse_args()


class FaceUVInference:
    """Class for running face UV coordinate estimation inference."""

    def __init__(
        self,
        checkpoint_path: str,
        dinov3_checkpoint_path: str = None,
        device: str = 'cuda',
        image_size: tuple = (512, 512),
        output_channels: int = 3,
    ):
        self.device = torch.device(device)
        self.image_size = image_size
        self.output_channels = output_channels

        # Load model
        self.model = self._load_model(checkpoint_path, dinov3_checkpoint_path)

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, checkpoint_path: str, dinov3_checkpoint_path: str = None):
        """Load model from checkpoint with separate dinov3 loading."""
        # Load dinov3 model first
        if dinov3_checkpoint_path:
            print(f"Loading DINOv3 from {dinov3_checkpoint_path}")
            REPO_DIR = "src/models/"
            dinov3 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local',
                                   weights=dinov3_checkpoint_path)
            dinov3 = dinov3.to(self.device)
        else:
            dinov3 = None

        # Load decoder checkpoint
        print(f"Loading decoder weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create model
        model = FaceUVModel(output_channels=self.output_channels)

        # Set dinov3 in encoder if provided
        if dinov3 is not None:
            model.encoder.dinov3 = dinov3
            model.encoder.dinov3.eval()
            for p in model.encoder.dinov3.parameters():
                p.requires_grad = False

        # Load state dict (decoder and CNN encoder weights)
        if 'state_dict' in checkpoint:
            # Filter out any remaining dinov3 keys if they exist
            state_dict = {k: v for k, v in checkpoint['state_dict'].items()
                         if 'encoder.dinov3' not in k}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.to(self.device)
        model.eval()

        return model

    def preprocess_image(self, image_path: str):
        """Preprocess input image."""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        # Apply transforms
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        return image_tensor, original_size

    def predict(self, image_paths):
        """
        Run inference on images for UV coordinate estimation.

        Args:
            image_paths: List of paths to input images

        Returns:
            Predicted UV maps and masks
        """
        # Preprocess images
        image_tensors = []
        original_sizes = []
        for image_path in image_paths:
            image_tensor, original_size = self.preprocess_image(image_path)
            image_tensors.append(image_tensor)
            original_sizes.append(original_size)

        image_tensors = torch.cat(image_tensors, dim=0)

        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensors)
            if isinstance(outputs, dict) and 'final' in outputs:
                pred_output = outputs['final']
            else:
                pred_output = outputs

            # Split UV and mask channels
            pred_uv = pred_output[:, :2]  # First two channels are U and V
            pred_mask = None
            if pred_output.shape[1] > 2:
                pred_mask = torch.sigmoid(pred_output[:, 2:])  # Third channel is mask

        return pred_uv, pred_mask, original_sizes

    def save_results(
        self,
        image_paths,
        pred_uv: torch.Tensor,
        pred_mask: Optional[torch.Tensor],
        original_sizes: list,
        output_dir: str,
        save: bool = True,
    ):
        """Save UV coordinate estimation results."""

        os.makedirs(output_dir, exist_ok=True)

        for img_idx, image_path in enumerate(image_paths):
            base_name = Path(image_path).stem

            # Get UV map for this image
            uv_map = pred_uv[img_idx]  # Shape: (2, H, W)

            # Resize to original size if needed
            if original_sizes[img_idx] is not None:
                uv_resized = torch.nn.functional.interpolate(
                    uv_map.unsqueeze(0),
                    size=original_sizes[img_idx][::-1],  # PIL uses (W, H), torch uses (H, W)
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                uv_resized = uv_map

            mask = pred_mask[img_idx].squeeze()
            if original_sizes[img_idx] is not None:
                mask_resized = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=original_sizes[img_idx][::-1],
                    mode='nearest'
                ).squeeze()
            else:
                mask_resized = mask

            # Convert UV map to numpy
            uv_np = uv_resized.cpu().numpy()  # Shape: (2, H, W)
            uv_np = np.transpose(uv_np, (1, 2, 0))  # Shape: (H, W, 2)

            # Denormalize UV coordinates from [-1, 1] to [0, 1]
            uv_np = (uv_np + 1.0) / 2.0
            uv_np = np.clip(uv_np, 0, 1)

            mask_np = (mask_resized.cpu().numpy() * 255).astype(np.uint8)

            if save:
                # Save UV map as 16-bit PNG (2 channels)
                # uv_16bit = (uv_np * 65535).astype(np.uint16)
                # cv2.imwrite(os.path.join(output_dir, f"{base_name}_uv.png"), uv_16bit)

                # Save mask
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), mask_np)

                # Visualize UV map as RGB (optional)
                uv_vis = np.zeros((uv_np.shape[0], uv_np.shape[1], 3), dtype=np.uint8)
                uv_vis[:, :, 0] = (uv_np[:, :, 0] * 255).astype(np.uint8)  # U -> Red
                uv_vis[:, :, 1] = (uv_np[:, :, 1] * 255).astype(np.uint8)  # V -> Green
                uv_vis[:, :, 2] = mask_np  # Mask -> Blue
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_uv_vis.png"), uv_vis)




    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        save: bool = True,
        batch_size = 1,
    ):
        """Process all images in a directory."""
        input_path = Path(input_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        if len(image_files) == 0:
            print(f"No images found in {input_dir}")
            return

        print(f"Found {len(image_files)} images to process")
        print(f"Processing with batch size {batch_size}")

        num_imgs = len(image_files)

        # Process each batch with progress tracking
        from tqdm import tqdm
        for img_idx in tqdm(range(0, num_imgs, batch_size), desc="Processing batches"):
            image_paths = image_files[img_idx:img_idx+batch_size]
            # try:
            pred_uv, pred_mask, original_sizes = self.predict(image_paths)
            self.save_results(
                image_paths,
                pred_uv,
                pred_mask,
                original_sizes,
                output_dir,
                save,
            )
            # except Exception as e:
            #     print(f"Error processing batch starting at index {img_idx}: {e}")
            #     continue



def main():
    """Main inference function."""
    args = parse_args()

    # Create inference object
    inference = FaceUVInference(
        checkpoint_path=args.checkpoint,
        dinov3_checkpoint_path=args.dinov3_checkpoint,
        device=args.device,
        image_size=tuple(args.image_size),
        output_channels=args.output_channels,
    )

    # Check if input is file or directory
    input_path = Path(args.input_dir)

    if input_path.is_file():
        # Single image inference
        print(f"Processing single image: {input_path}")
        pred_uv, pred_mask, original_sizes = inference.predict([str(input_path)])
        inference.save_results(
            [str(input_path)],
            pred_uv,
            pred_mask,
            original_sizes,
            args.output_dir,
            args.save,

        )
        print(f"Results saved to {args.output_dir}")
    elif input_path.is_dir():
        # Process directory
        inference.process_directory(
            args.input_dir,
            args.output_dir,
            args.save,
            batch_size=args.batch_size,
        )
        print(f"Processing complete. Results saved to {args.output_dir}")
    else:
        raise ValueError(f"Input path {input_path} does not exist")



if __name__ == '__main__':
    main()
