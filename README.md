# Blur Daddy

Blur Daddy is an open-source tool designed to enable precise face-blurring in photos and videos. This project aims to provide a comprehensive solution for privacy protection in media by offering various blurring techniques and advanced features.

## Features (so far)

- **Face Blurring in Static Images**: Our initial release supports blurring faces in static images using Gaussian blur and pixelation techniques.


## Milestones

1. ✅ **Milestone 1: MVP - Blur Faces in Static Images**
   - Implemented face detection and blurring in static images.
   - Supports Gaussian blur and pixelation methods.

2. ⬜ **Milestone 2: Video Blurring Support (Frame-by-Frame)**
   - Extend functionality to support video files, processing each frame individually.

3. ⬜ **Milestone 3: Face Tracking & ID Assignment for Videos**
   - Implement face tracking across video frames with ID assignment for consistent blurring.

4. TBD

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV
- Other dependencies as listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ashkix/blur-daddy.git
   cd blur-daddy
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To blur faces in an image, run the following command:

```bash
python main/blur_faces.py --input path/to/your/image.jpg --output path/to/save/blurred_image.jpg --method gaussian
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
