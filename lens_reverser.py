import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


class PhoneLensReverser:
    """
    Reverses common phone camera lens distortions and creates 3D perspective views.
    Handles barrel, pincushion, and fisheye distortions common in smartphone cameras.
    """
    
    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.original = None
        self.height = 0
        self.width = 0
        self.center_x = 0
        self.center_y = 0
        self.load_image()
    
    def load_image(self):
        """Load and validate the input image."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        self.original = cv2.imread(str(self.image_path))
        if self.original is None:
            raise ValueError(f"Could not load image: {self.image_path}")
        
        self.height, self.width = self.original.shape[:2]
        self.center_x = self.width / 2
        self.center_y = self.height / 2
        print(f"Loaded image: {self.width}x{self.height}")

    def barrel_distortion_correction(self, strength=0.5, zoom=1.0):
        """
        Corrects barrel distortion (bulging outward) common in wide-angle phone lenses.
        Uses the inverse radial distortion model.
        
        Args:
            strength: Correction strength (0.0 to 2.0)
            zoom: Zoom factor to crop empty edges after correction
        """
        # Create normalized coordinate grids
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xv, yv = np.meshgrid(x, y)
        
        # Calculate radial distance from center
        r = np.sqrt(xv**2 + yv**2)
        
        # Barrel distortion correction: r_corrected = r * (1 + k * r^2)
        # We apply the inverse to undistort
        k = -strength * 0.5  # Negative for correction
        r_corrected = r * (1 + k * r**2 + k * 0.5 * r**4)
        
        # Avoid division by zero and extreme values
        r_corrected = np.clip(r_corrected, 0.001, 2.0)
        
        # Calculate scaling factor
        scale = np.where(r > 0, r_corrected / r, 1.0)
        
        # Apply scaling to coordinates
        xv_new = xv * scale
        yv_new = yv * scale
        
        # Convert back to pixel coordinates
        map_x = ((xv_new + 1) / 2 * self.width).astype(np.float32)
        map_y = ((yv_new + 1) / 2 * self.height).astype(np.float32)
        
        # Remap the image
        corrected = cv2.remap(self.original, map_x, map_y, 
                             interpolation=cv2.INTER_LANCZOS4, 
                             borderMode=cv2.BORDER_CONSTANT)
        
        # Apply zoom to crop empty edges
        if zoom != 1.0:
            corrected = self._apply_zoom(corrected, zoom)
        
        return corrected
    
    def fisheye_correction(self, fov=180, zoom=1.2):
        """
        Corrects fisheye distortion using equidistant projection model.
        Common in ultra-wide phone cameras.
        
        Args:
            fov: Field of view in degrees
            zoom: Zoom factor to fill frame after correction
        """
        # Convert FOV to radians
        fov_rad = np.radians(fov)
        
        # Create coordinate grids
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xv, yv = np.meshgrid(x, y)
        
        # Calculate normalized radius and angle
        r_norm = np.sqrt(xv**2 + yv**2)
        theta = np.arctan2(yv, xv)
        
        # Avoid division by zero
        r_norm = np.clip(r_norm, 0.0001, 1.0)
        
        # Fisheye equidistant projection: r = f * theta
        # To correct: theta = r / f, where f is focal length in pixels
        max_r = np.sqrt(2)  # Maximum normalized radius
        f = max_r / (fov_rad / 2)  # Focal length
        
        # Calculate angle from center for each pixel
        theta_dist = r_norm * (fov_rad / 2)
        
        # Clip to valid range
        theta_dist = np.clip(theta_dist, -np.pi/2 + 0.01, np.pi/2 - 0.01)
        
        # Apply perspective (pinhole) projection: r_perspective = f * tan(theta)
        r_perspective = f * np.tan(theta_dist)
        
        # Scale back to normalized coordinates
        scale = np.where(r_norm > 0, r_perspective / r_norm, 1.0)
        
        xv_new = xv * scale
        yv_new = yv * scale
        
        # Convert to pixel coordinates
        map_x = ((xv_new + 1) / 2 * self.width).astype(np.float32)
        map_y = ((yv_new + 1) / 2 * self.height).astype(np.float32)
        
        # Remap
        corrected = cv2.remap(self.original, map_x, map_y,
                             interpolation=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_CONSTANT)
        
        if zoom != 1.0:
            corrected = self._apply_zoom(corrected, zoom)
        
        return corrected
    
    def perspective_3d_view(self, rotation_x=15, rotation_y=15, translation_z=0, 
                           fov=60, output_size=None):
        """
        Creates a 3D perspective view of the image.
        Simulates viewing the image from an angle in 3D space.
        
        Args:
            rotation_x: Rotation around X-axis in degrees (tilt up/down)
            rotation_y: Rotation around Y-axis in degrees (pan left/right)
            translation_z: Translation along Z-axis (zoom in/out)
            fov: Field of view in degrees
            output_size: (width, height) of output, defaults to input size
        """
        if output_size is None:
            output_size = (self.width, self.height)
        
        # Convert rotations to radians
        rx = np.radians(rotation_x)
        ry = np.radians(rotation_y)
        
        # Intrinsic camera matrix
        focal_length = self.width / (2 * np.tan(np.radians(fov) / 2))
        cx, cy = self.width / 2, self.height / 2
        
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ], dtype=np.float64)
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ], dtype=np.float64)
        
        # Combined rotation
        R = Ry @ Rx
        
        # Translation vector
        t = np.array([[0], [0], [translation_z]], dtype=np.float64)
        
        # Create 3D plane points (image at z=0)
        # We create a grid of points on the image plane
        x = np.linspace(0, self.width - 1, self.width)
        y = np.linspace(0, self.height - 1, self.height)
        xv, yv = np.meshgrid(x, y)
        
        # Flatten for matrix operations
        points_3d = np.vstack([
            xv.flatten() - cx,
            yv.flatten() - cy,
            np.zeros(self.width * self.height)
        ])
        
        # Apply rotation and translation
        points_transformed = R @ points_3d + t
        
        # Project to 2D
        # Avoid division by zero or negative z
        z = points_transformed[2, :]
        z = np.clip(z, 0.1, None)
        
        x_proj = (points_transformed[0, :] / z) * focal_length + cx
        y_proj = (points_transformed[1, :] / z) * focal_length + cy
        
        # Reshape to image dimensions
        map_x = x_proj.reshape(self.height, self.width).astype(np.float32)
        map_y = y_proj.reshape(self.height, self.width).astype(np.float32)
        
        # Remap
        result = cv2.remap(self.original, map_x, map_y,
                          interpolation=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))
        
        # Resize if needed
        if output_size != (self.width, self.height):
            result = cv2.resize(result, output_size, interpolation=cv2.INTER_LANCZOS4)
        
        return result
    
    def cylindrical_projection(self, fov=120):
        """
        Projects the image onto a cylindrical surface.
        Useful for creating panoramic 3D-like views.
        
        Args:
            fov: Horizontal field of view in degrees
        """
        fov_rad = np.radians(fov)
        f = self.width / (2 * np.tan(fov_rad / 2))
        
        # Create coordinate grids
        x = np.arange(self.width)
        y = np.arange(self.height)
        xv, yv = np.meshgrid(x, y)
        
        # Normalize x to angle
        theta = (xv - self.center_x) / f
        
        # Clip to valid range
        theta = np.clip(theta, -fov_rad/2, fov_rad/2)
        
        # Cylindrical projection
        x_cyl = f * np.tan(theta) + self.center_x
        y_cyl = (yv - self.center_y) / np.cos(theta) + self.center_y
        
        map_x = x_cyl.astype(np.float32)
        map_y = y_cyl.astype(np.float32)
        
        result = cv2.remap(self.original, map_x, map_y,
                          interpolation=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT)
        
        return result
    
    def stereographic_projection(self, fov=180):
        """
        Applies stereographic projection for extreme wide-angle correction.
        Maps sphere to plane, preserving angles.
        """
        fov_rad = np.radians(fov)
        
        # Create normalized coordinates
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xv, yv = np.meshgrid(x, y)
        
        # Calculate spherical coordinates
        r = np.sqrt(xv**2 + yv**2)
        theta = np.arctan2(yv, xv)
        
        # Clip radius
        r = np.clip(r, 0.0001, 1.0)
        
        # Stereographic projection from sphere to plane
        # Angle from optical axis
        phi = r * (fov_rad / 2)
        phi = np.clip(phi, 0, np.pi - 0.01)
        
        # Stereographic mapping
        r_stereo = 2 * np.tan(phi / 2)
        r_stereo = np.clip(r_stereo, 0, 10)
        
        # Scale to fill image
        scale = np.where(r > 0, r_stereo / (2 * r), 1.0)
        
        xv_new = xv * scale
        yv_new = yv * scale
        
        # Convert to pixel coordinates
        map_x = ((xv_new + 1) / 2 * self.width).astype(np.float32)
        map_y = ((yv_new + 1) / 2 * self.height).astype(np.float32)
        
        result = cv2.remap(self.original, map_x, map_y,
                          interpolation=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT)
        
        return result
    
    def _apply_zoom(self, image, zoom_factor):
        """Apply zoom by cropping and resizing."""
        h, w = image.shape[:2]
        
        # Calculate crop dimensions
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        
        # Calculate crop coordinates
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        x2 = x1 + new_w
        y2 = y1 + new_h
        
        # Crop and resize back
        cropped = image[y1:y2, x1:x2]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    def compare_all(self, save_path=None):
        """
        Generate a comparison image showing original and all corrections.
        """
        results = {
            'Original': self.original,
            'Barrel Correction': self.barrel_distortion_correction(strength=0.6, zoom=1.15),
            'Fisheye Correction': self.fisheye_correction(fov=170, zoom=1.3),
            '3D Perspective': self.perspective_3d_view(rotation_x=20, rotation_y=15, fov=70),
            'Cylindrical': self.cylindrical_projection(fov=120),
        }
        
        # Create figure with subplots
        n = len(results)
        cols = 3
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if n > 1 else [axes]
        
        for idx, (title, img) in enumerate(results.items()):
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img_rgb)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")
        
        plt.show()
        
        return results
    
    def save_image(self, image, output_path):
        """Save processed image to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        print(f"Saved: {output_path}")


def find_images(directory="image"):
    """Find all image files in the given directory."""
    image_dir = Path(directory)
    if not image_dir.exists():
        return []
    
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = [f for f in image_dir.iterdir() 
              if f.is_file() and f.suffix.lower() in extensions]
    return sorted(images)


def interactive_mode(reverser):
    """Interactive mode for fine-tuning corrections."""
    print("\n=== Interactive Phone Lens Reverser ===")
    print("Controls (OpenCV window):")
    print("  'b' - Barrel distortion correction")
    print("  'f' - Fisheye correction")
    print("  'p' - 3D Perspective view")
    print("  'c' - Cylindrical projection")
    print("  's' - Stereographic projection")
    print("  'o' - Original image")
    print("  'q' - Quit")
    print("  '+'/'-' - Increase/decrease strength")
    print("  Arrow keys - Adjust 3D view angles")
    print("  'r' - Reset parameters")
    print("  'S' - Save current view")
    
    current_mode = 'o'
    strength = 0.5
    zoom = 1.0
    rot_x, rot_y = 0, 0
    fov = 120
    
    # Resize for display if too large
    display_scale = 1.0
    max_display = 1000
    if reverser.width > max_display or reverser.height > max_display:
        display_scale = max_display / max(reverser.width, reverser.height)
    
    display_w = int(reverser.width * display_scale)
    display_h = int(reverser.height * display_scale)
    
    while True:
        # Generate current view
        if current_mode == 'o':
            display = reverser.original.copy()
            title = "Original"
        elif current_mode == 'b':
            display = reverser.barrel_distortion_correction(strength=strength, zoom=zoom)
            title = f"Barrel Correction (s={strength:.2f}, z={zoom:.2f})"
        elif current_mode == 'f':
            display = reverser.fisheye_correction(fov=fov, zoom=zoom)
            title = f"Fisheye Correction (fov={fov:.0f}, z={zoom:.2f})"
        elif current_mode == 'p':
            display = reverser.perspective_3d_view(rotation_x=rot_x, rotation_y=rot_y, fov=70)
            title = f"3D Perspective (rx={rot_x:.0f}, ry={rot_y:.0f})"
        elif current_mode == 'c':
            display = reverser.cylindrical_projection(fov=fov)
            title = f"Cylindrical (fov={fov:.0f})"
        elif current_mode == 's':
            display = reverser.stereographic_projection(fov=fov)
            title = f"Stereographic (fov={fov:.0f})"
        else:
            display = reverser.original.copy()
            title = "Unknown"
        
        # Resize for display
        if display_scale < 1.0:
            display = cv2.resize(display, (display_w, display_h))
        
        # Add title text
        cv2.putText(display, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press 'h' for help", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 200, 0), 1)
        
        cv2.imshow('Phone Lens Reverser', display)
        
        key = cv2.waitKey(50) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('o'):
            current_mode = 'o'
        elif key == ord('b'):
            current_mode = 'b'
        elif key == ord('f'):
            current_mode = 'f'
        elif key == ord('p'):
            current_mode = 'p'
        elif key == ord('c'):
            current_mode = 'c'
        elif key == ord('s'):
            current_mode = 's'
        elif key == ord('h'):
            print("\nControls:")
            print("  b/f/p/c/s/o - Switch modes")
            print("  +/- - Adjust strength/zoom")
            print("  arrows - Adjust 3D rotation")
            print("  [/] - Adjust FOV")
            print("  r - Reset")
            print("  S - Save")
            print("  q - Quit")
        elif key == ord('+') or key == ord('='):
            if current_mode in ['b']:
                strength = min(strength + 0.1, 2.0)
            elif current_mode in ['f', 'c', 's']:
                fov = min(fov + 10, 200)
            elif current_mode == 'p':
                rot_x = min(rot_x + 5, 45)
            else:
                zoom = min(zoom + 0.1, 2.0)
        elif key == ord('-') or key == ord('_'):
            if current_mode in ['b']:
                strength = max(strength - 0.1, 0.1)
            elif current_mode in ['f', 'c', 's']:
                fov = max(fov - 10, 60)
            elif current_mode == 'p':
                rot_x = max(rot_x - 5, -45)
            else:
                zoom = max(zoom - 0.1, 1.0)
        elif key == 81:  # Left arrow
            if current_mode == 'p':
                rot_y = max(rot_y - 5, -45)
        elif key == 83:  # Right arrow
            if current_mode == 'p':
                rot_y = min(rot_y + 5, 45)
        elif key == 82:  # Up arrow
            if current_mode == 'p':
                rot_x = min(rot_x + 5, 45)
        elif key == 84:  # Down arrow
            if current_mode == 'p':
                rot_x = max(rot_x - 5, -45)
        elif key == ord('['):
            zoom = max(zoom - 0.05, 1.0)
        elif key == ord(']'):
            zoom = min(zoom + 0.05, 2.0)
        elif key == ord('r'):
            strength = 0.5
            zoom = 1.0
            rot_x, rot_y = 0, 0
            fov = 120
        elif key == ord('S'):
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            timestamp = cv2.getTickCount()
            filename = f"{current_mode}_{timestamp}.jpg"
            output_path = output_dir / filename
            
            # Save full resolution
            if current_mode == 'o':
                save_img = reverser.original
            elif current_mode == 'b':
                save_img = reverser.barrel_distortion_correction(strength, zoom)
            elif current_mode == 'f':
                save_img = reverser.fisheye_correction(fov, zoom)
            elif current_mode == 'p':
                save_img = reverser.perspective_3d_view(rot_x, rot_y, fov=70)
            elif current_mode == 'c':
                save_img = reverser.cylindrical_projection(fov)
            elif current_mode == 's':
                save_img = reverser.stereographic_projection(fov)
            else:
                save_img = reverser.original
            
            reverser.save_image(save_img, output_path)
    
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Phone Lens Reverser - Correct lens distortions and create 3D views'
    )
    parser.add_argument('--image', '-i', type=str, default=None,
                       help='Path to input image')
    parser.add_argument('--mode', '-m', type=str, default='interactive',
                       choices=['interactive', 'barrel', 'fisheye', 'perspective', 
                               'cylindrical', 'stereographic', 'compare', 'all'],
                       help='Processing mode')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--strength', '-s', type=float, default=0.5,
                       help='Correction strength (0.0-2.0)')
    parser.add_argument('--zoom', '-z', type=float, default=1.0,
                       help='Zoom factor')
    parser.add_argument('--fov', type=float, default=120,
                       help='Field of view in degrees')
    parser.add_argument('--rot-x', type=float, default=15,
                       help='X rotation for perspective mode')
    parser.add_argument('--rot-y', type=float, default=15,
                       help='Y rotation for perspective mode')
    
    args = parser.parse_args()
    
    # Find image
    if args.image:
        image_path = args.image
    else:
        images = find_images("image")
        if not images:
            print("No images found in ./image directory!")
            print("Usage: Place images in ./image folder or specify with --image")
            print("\nCreating image directory...")
            Path("image").mkdir(exist_ok=True)
            print("Please add images to the 'image' folder and run again.")
            return
        image_path = str(images[0])
        print(f"Found image: {image_path}")
    
    # Initialize reverser
    reverser = PhoneLensReverser(image_path)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process based on mode
    if args.mode == 'interactive':
        interactive_mode(reverser)
    
    elif args.mode == 'barrel':
        result = reverser.barrel_distortion_correction(args.strength, args.zoom)
        output_path = output_dir / f"barrel_corrected.jpg"
        reverser.save_image(result, output_path)
    
    elif args.mode == 'fisheye':
        result = reverser.fisheye_correction(args.fov, args.zoom)
        output_path = output_dir / f"fisheye_corrected.jpg"
        reverser.save_image(result, output_path)
    
    elif args.mode == 'perspective':
        result = reverser.perspective_3d_view(args.rot_x, args.rot_y, fov=args.fov)
        output_path = output_dir / f"perspective_3d.jpg"
        reverser.save_image(result, output_path)
    
    elif args.mode == 'cylindrical':
        result = reverser.cylindrical_projection(args.fov)
        output_path = output_dir / f"cylindrical.jpg"
        reverser.save_image(result, output_path)
    
    elif args.mode == 'stereographic':
        result = reverser.stereographic_projection(args.fov)
        output_path = output_dir / f"stereographic.jpg"
        reverser.save_image(result, output_path)
    
    elif args.mode == 'compare':
        output_path = output_dir / "comparison.png"
        reverser.compare_all(save_path=output_path)
    
    elif args.mode == 'all':
        # Generate all corrections
        corrections = {
            'barrel': reverser.barrel_distortion_correction(args.strength, args.zoom),
            'fisheye': reverser.fisheye_correction(args.fov, args.zoom),
            'perspective': reverser.perspective_3d_view(args.rot_x, args.rot_y, fov=args.fov),
            'cylindrical': reverser.cylindrical_projection(args.fov),
            'stereographic': reverser.stereographic_projection(args.fov),
        }
        
        for name, img in corrections.items():
            output_path = output_dir / f"{name}.jpg"
            reverser.save_image(img, output_path)
        
        # Also save comparison
        comparison_path = output_dir / "comparison.png"
        reverser.compare_all(save_path=comparison_path)
    
    print(f"\nDone! Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()

