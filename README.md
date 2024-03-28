# AI-powered-Accessibility-for-Visually-Impaired
Entwickeln Sie eine KI-gestützte Anwendung, die Bilder und Videos für Sehbehinderte beschreibt und so die Barrierefreiheit digitaler Inhalte verbessert.
from transformers import pipeline
from PIL import Image
import cv2

# Initialize the image-to-text pipeline with a pre-trained model
image_description_pipeline = pipeline('image-to-text')

def describe_image(image_path):
    """
    Generates a description for an image.
    
    Parameters:
    - image_path: The path to the image file.
    
    Returns:
    A string containing the description of the image.
    """
    image = Image.open(image_path)
    description = image_description_pipeline(image)[0]['generated_text']
    return description

def extract_key_frames(video_path):
    """
    Extracts key frames from a video for description. This is a simplified approach,
    focusing on extracting frames at a regular interval.
    
    Parameters:
    - video_path: The path to the video file.
    
    Returns:
    A list of PIL Image objects representing key frames from the video.
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frames = []
    frame_rate = int(vidcap.get(cv2.CAP_PROP_FPS)) * 5  # Example: One frame every 5 seconds
    
    while success:
        frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, vidcap.get(cv2.CAP_PROP_POS_FRAMES) + frame_rate)
        success, image = vidcap.read()
    
    vidcap.release()
    return frames

def describe_video(video_path):
    """
    Generates descriptions for key frames in a video.
    
    Parameters:
    - video_path: The path to the video file.
    
    Returns:
    A list of strings, each containing the description of a key frame.
    """
    key_frames = extract_key_frames(video_path)
    descriptions = [describe_image(frame) for frame in key_frames]
    return descriptions

# Example usage
image_description = describe_image('path/to/your/image.jpg')
print("Image Description:", image_description)

video_descriptions = describe_video('path/to/your/video.mp4')
print("Video Descriptions:")
for desc in video_descriptions:
    print(desc)
