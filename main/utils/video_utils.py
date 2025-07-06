import cv2

def extract_frames(video_path: str):
    """
    Extract frames from a video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    return frames

def get_video_metadata(video_path: str):
    """
    Get metadata from a video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, (width, height)

def write_video(frames: list, output_path: str, fps: int, size: tuple):
    """
    Write frames to a video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    for frame in frames:
        out.write(frame)
    out.release()
