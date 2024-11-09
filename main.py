import os
import numpy as np
from deepface import DeepFace
import faiss
import cv2
import io
import uuid
from PIL import Image
import requests
import base64
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import pickle
from dotenv import load_dotenv
import traceback  # Import traceback for detailed error logging
import urllib.request

app = FastAPI()

load_dotenv()

# Directory to store images and embeddings
IMAGE_DIR = os.environ.get('IMAGE_DIR')
EMBEDDING_DIR = os.environ.get('EMBEDDING_DIR')
print(IMAGE_DIR, EMBEDDING_DIR)
INDEX_FILE = os.path.join(EMBEDDING_DIR, "faiss_index.bin")
MAPPING_FILE = os.path.join(EMBEDDING_DIR, "vector_to_image_map.pkl")
THRESHOLD = 0.7 # Adjust this threshold as needed

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Initialize Faiss index
# dimension = 2622  # DeepFace embedding dimension
dimension = 128
index = None
vector_to_image_map = {}


def get_embedding(img_path):
    try:
        embeddings = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)

        if isinstance(embeddings, list) and len(embeddings) > 0:
            embedding = embeddings[0].get('embedding', None)
            model_info = embeddings[0].get('model', 'Facenet')  # Use 'Facenet' as default
        elif isinstance(embeddings, dict):
            embedding = embeddings.get('embedding', None)
            model_info = embeddings.get('model', 'Facenet')  # Use 'Facenet' as default
        else:
            raise ValueError(f"Unexpected embeddings structure for {img_path}")

        print(f"Model used: {model_info}")

        if embedding is None:
            raise ValueError(f"No embedding found for {img_path}")
        
        return np.array(embedding).astype('float32')

    except Exception as e:
        print(f"Error in get_embedding: {str(e)}")
        raise

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def save_index_and_mapping():
    faiss.write_index(index, INDEX_FILE)
    with open(MAPPING_FILE, 'wb') as f:
        pickle.dump(vector_to_image_map, f)

def load_index_and_mapping():
    global index, vector_to_image_map
    if os.path.exists(INDEX_FILE) and os.path.exists(MAPPING_FILE):
        try:
            temp_index = faiss.read_index(INDEX_FILE)
            if temp_index.d == dimension:
                index = temp_index
                with open(MAPPING_FILE, 'rb') as f:
                    vector_to_image_map = pickle.load(f)
                print(f"Loaded existing index with dimension {dimension}")
            else:
                print(f"Existing index has incorrect dimension. Creating new index.")
                index = faiss.IndexFlatIP(dimension)  # Initialize new index
                vector_to_image_map = {}
        except Exception as e:
            print(f"Error loading index: {str(e)}. Creating new index.")
            index = faiss.IndexFlatIP(dimension)  # Initialize new index if load fails
            vector_to_image_map = {}
    else:
        print("No existing index found. Creating new index.")
        index = faiss.IndexFlatIP(dimension)  # Initialize new index
        vector_to_image_map = {}


def process_image_folder():
    print(f"Faiss index dimension: {index.d}")
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(IMAGE_DIR, filename)
            if img_path not in vector_to_image_map.values():
                try:
                    print(f"Processing {filename}...")
                    embedding = get_embedding(img_path)
                    print(f"Embedding shape: {embedding.shape}")

                    normalized_embedding = normalize_vector(embedding)
                    index.add(normalized_embedding.reshape(1, -1))
                    vector_to_image_map[index.ntotal - 1] = img_path
                    print(f"Successfully processed {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    traceback.print_exc()
                    print(f"Skipping {filename} due to error")
                    continue

    save_index_and_mapping()


import logging

logging.basicConfig(level=logging.DEBUG)


class ImageRequest(BaseModel):
    image_url: str

@app.post("/compare_image")
async def compare_image(image_request: ImageRequest):
    image_url = image_request.image_url
    temp_image_path = os.path.join(IMAGE_DIR, "temp_search_image.jpg")
    is_matched = True

    try:
        # Download and save the image using urllib
        try:
            urllib.request.urlretrieve(image_url, temp_image_path)
            logging.info(f"Image downloaded and saved to {temp_image_path}")
        except Exception as e:
            logging.error(f"Failed to download image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error downloading image: {str(e)}")

        # Use OpenCV to read the saved image
        img = cv2.imread(temp_image_path)
        
        if img is None:
            raise ValueError("Downloaded image could not be opened.")
        
        # Get embedding for the search image
        search_embedding = get_embedding(temp_image_path)
        
        # Convert to numpy array and normalize
        search_embedding = np.array(search_embedding, dtype=np.float32)
        search_embedding_norm = np.linalg.norm(search_embedding)
        if search_embedding_norm != 0:
            normalized_search_embedding = search_embedding / search_embedding_norm
        else:
            normalized_search_embedding = search_embedding

        # Get all stored embeddings for batch comparison
        if index.ntotal == 0:
            raise ValueError("No images in the database to compare against.")

        # Perform batch similarity computation using FAISS
        similarities, indices = index.search(
            normalized_search_embedding.reshape(1, -1), 
            index.ntotal
        )

        # Use vectorized operations to find matches above threshold
        mask = similarities[0] > THRESHOLD
        if not np.any(mask):
            is_matched = False
            raise HTTPException(status_code=404, detail="No matching image found above the threshold")

        # Get the best match
        best_idx = np.argmax(similarities[0])
        best_similarity = float(similarities[0][best_idx])
        matched_image_path = vector_to_image_map[int(indices[0][best_idx])]

        return {
            "is_matched": is_matched,
            "matched_image_path": matched_image_path,
            # "similarity": best_similarity
        }

    except ValueError as e:
        logging.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@app.post("/store_image")
async def store_image(image_request: ImageRequest, image_path: str = Body(...)):
    image_url = image_request.image_url

    try:
        # Check if an image with the same base filename exists
        base_filename = os.path.splitext(image_path)[0]
        existing_indices = [idx for idx, path in vector_to_image_map.items() 
                            if os.path.splitext(os.path.basename(path))[0] == base_filename]

        # If a matching image exists, replace it
        if existing_indices:
            for idx in existing_indices:
                existing_image_path = vector_to_image_map[idx]
                logging.info(f"Replacing existing image at {existing_image_path}")

                # Remove the old image file
                if os.path.exists(existing_image_path):
                    os.remove(existing_image_path)
                    logging.info(f"Removed old image file: {existing_image_path}")

                # Remove the old embedding from the FAISS index
                index.remove_ids(np.array([idx], dtype=np.int64))

                # Remove the entry from the mapping
                del vector_to_image_map[idx]

        # Download the new image with urllib
        try:
            req = urllib.request.Request(image_url, headers={
                'User-Agent': 'Mozilla/5.0',
                'Accept': '*/*'
            })
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to download image. HTTP Status: {response.status}")
                image_content = response.read()
        except Exception as e:
            logging.error(f"Error downloading image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error downloading image: {str(e)}")

        # Try to open the image content with Pillow
        try:
            with Image.open(io.BytesIO(image_content)) as img:
                # If we can open it with Pillow, it's a valid image
                # Save the image
                full_image_path = os.path.join(IMAGE_DIR, image_path)
                os.makedirs(os.path.dirname(full_image_path), exist_ok=True)
                img.save(full_image_path)
                
                # Get the actual image format
                img_format = img.format.lower() if img.format else "unknown"
                logging.info(f"Successfully validated image format: {img_format}")
        except Exception as img_error:
            # If Pillow can't open it, then check the content type as a fallback
            raise ValueError(f"Failed to process image: {str(img_error)}")

        # Confirm the image was saved successfully
        if not os.path.exists(full_image_path):
            raise FileNotFoundError(f"Image file {full_image_path} was not saved.")

        # Get embedding for the new image
        embedding = get_embedding(full_image_path)
        normalized_embedding = normalize_vector(embedding)

        # Add the new embedding to the FAISS index
        index.add(normalized_embedding.reshape(1, -1))

        # Update vector_to_image_map with the new image
        new_index = index.ntotal - 1
        vector_to_image_map[new_index] = full_image_path

        # Save updated index and mapping
        save_index_and_mapping()

        return {
            "message": "Image stored successfully",
            "image_path": full_image_path,
            "index": new_index,
            "format": img_format if 'img_format' in locals() else "unknown"
        }

    except ValueError as e:
        logging.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

@app.post("/start_processing")
async def start_processing():
    try:
        print("Loading existing index and mappings...")
        load_index_and_mapping()
        print("Processing image folder...")
        process_image_folder()
        print("Image processing complete. Starting server...")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

def initialize_index():
    """Initialize or load existing FAISS index"""
    global index, vector_to_image_map
    
    try:
        if os.path.exists(INDEX_FILE) and os.path.exists(MAPPING_FILE):
            # Load existing index
            temp_index = faiss.read_index(INDEX_FILE)
            if temp_index.d == dimension:
                index = temp_index
                with open(MAPPING_FILE, 'rb') as f:
                    vector_to_image_map = pickle.load(f)
                logging.info(f"Loaded existing index with dimension {dimension}")
            else:
                logging.warning("Existing index has incorrect dimension. Creating new index.")
                index = faiss.IndexFlatIP(dimension)
                vector_to_image_map = {}
        else:
            logging.info("No existing index found. Creating new index.")
            index = faiss.IndexFlatIP(dimension)
            vector_to_image_map = {}
    except Exception as e:
        logging.error(f"Error initializing index: {str(e)}")
        index = faiss.IndexFlatIP(dimension)
        vector_to_image_map = {}

    return index is not None
    
@app.on_event("startup")
async def startup_event():
    initialize_index()


if __name__ == '__main__':
    app.run(host = '0.0.0.0',debug=True)