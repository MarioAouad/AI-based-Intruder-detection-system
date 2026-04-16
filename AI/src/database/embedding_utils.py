import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from deepface import DeepFace

def get_embedding(image_path: str) -> list[float]:
    """
    Reads an aligned image from disk and converts it into a 512-D mathematical embedding.
    """
    # enforce_detection=False prevents crashes if the alignment made it slightly blurry
    result = DeepFace.represent(
        img_path=image_path, 
        model_name="Facenet512", 
        enforce_detection=False
    )
    
    # Return the actual embedding array for the first face found
    return result[0]["embedding"]