# Phone Lens Reverser

A Python tool that mathematically reverses phone camera lens distortions and creates 3D perspective views from 2D images.

## Features

- **Barrel Distortion Correction**: Fixes bulging outward distortion common in wide-angle phone lenses
- **Fisheye Correction**: Corrects ultra-wide angle fisheye distortion using equidistant projection model
- **3D Perspective View**: Simulates viewing the image from different angles in 3D space
- **Cylindrical Projection**: Projects images onto a cylindrical surface for panoramic views
- **Stereographic Projection**: Corrects extreme wide-angle distortion while preserving angles
- **Interactive Mode**: Real-time parameter tuning with keyboard controls
- **Batch Processing**: Generate all corrections at once

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode (Recommended)

Place your images in the `/image` folder, then run:

```bash
python lens_reverser.py
```

**Controls:**
- `b` - Barrel distortion correction
- `f` - Fisheye correction
- `p` - 3D Perspective view
- `c` - Cylindrical projection
- `s` - Stereographic projection
- `o` - Original image
- `+`/`-` - Increase/decrease strength
- `Arrow keys` - Adjust 3D view angles (in perspective mode)
- `[`/`]` - Adjust zoom
- `r` - Reset parameters
- `S` - Save current view
- `q` - Quit

### Command Line Modes

```bash
# Correct barrel distortion
python lens_reverser.py --mode barrel --strength 0.6 --zoom 1.15

# Correct fisheye distortion
python lens_reverser.py --mode fisheye --fov 170 --zoom 1.3

# Create 3D perspective view
python lens_reverser.py --mode perspective --rot-x 20 --rot-y 15

# Generate all corrections + comparison
python lens_reverser.py --mode all

# Generate comparison grid only
python lens_reverser.py --mode compare
```

### Specify Custom Image

```bash
python lens_reverser.py --image path/to/your/image.jpg --mode barrel
```

## Mathematical Models

### Barrel Distortion Correction
Uses the inverse radial distortion model:
```
r_corrected = r * (1 + k1*r² + k2*r⁴)
```
Where negative coefficients correct barrel distortion.

### Fisheye Correction
Converts equidistant projection to perspective projection:
```
r_fisheye = f * θ
r_perspective = f * tan(θ)
```

### 3D Perspective Transform
Applies rotation matrices Rx, Ry and perspective projection:
```
[x']   [focal_length  0      cx  ]   [X/Z]
[y'] = [0      focal_length  cy  ] * [Y/Z]
[1 ]   [0           0        1   ]   [ 1 ]
```

## Output

All processed images are saved to the `/output` directory.

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- SciPy
- Pillow

