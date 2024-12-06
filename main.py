import os
import numpy as np
from deepface import DeepFace
import faiss
import cv2
import logging
from PIL import Image
from scipy.spatial.distance import cosine
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import shutil
import pickle
from dotenv import load_dotenv
import urllib.request
from botocore.exceptions import ClientError
from fastapi import HTTPException
from s3_helper import S3Helper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Constants
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
S3_IMAGE_FOLDER = os.environ.get('S3_IMAGE_FOLDER', 'images/')
EMBEDDING_DIR = os.environ.get('EMBEDDING_DIR')
INDEX_FILE = os.path.join(EMBEDDING_DIR, "faiss_index.bin")
MAPPING_FILE = os.path.join(EMBEDDING_DIR, "vector_to_image_map.pkl")
SIMILARITY_THRESHOLD = 0.85
DIMENSION = 512
EMBEDDING_MODEL = "FaceNet512"

# Ensure directories exist
os.makedirs(EMBEDDING_DIR, exist_ok=True)

index = None
vector_to_image_map = {}

class ImageRequest(BaseModel):
    image_url: str

class FaceProcessor:
    def __init__(self, min_face_size=(160, 160)):
        self.min_face_size = min_face_size
        self.detection_backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']
        
    def detect_and_crop_face(self, image_path):
        print(image_path, "---------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!---------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for backend in self.detection_backends:
            try:
                # Read the original image
                original_img = cv2.imread(image_path)
                if original_img is None:
                    raise ValueError(f"Could not read image from {image_path}")

                # Detect face using DeepFace
                face_objs = DeepFace.extract_faces(
                    img_path=image_path,
                    detector_backend=backend,
                    enforce_detection=True,
                    align=True
                )

                if not face_objs:
                    continue

                # Sort faces by confidence if multiple detected
                face_objs.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                face_obj = face_objs[0]  # Take most confident face
                
                if face_obj.get('confidence', 0) < 0.9:
                    continue

                # Get facial area
                facial_area = face_obj['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

                # Crop face from original image
                face_img = original_img[y:y+h, x:x+w]

                # Resize to target size if needed
                if face_img.shape[:2] != self.min_face_size:
                    face_img = cv2.resize(face_img, self.min_face_size)
                
                # Save cropped face
                working_directory = os.getcwd()  # or specify your desired directory
                cropped_path = os.path.join(working_directory, f"cropped_face_{backend}.jpg")
                cv2.imwrite(cropped_path, face_img)

                # Debug: Save facial area outline on original image
                debug_img = original_img.copy()
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                debug_path = os.path.join(working_directory, f"debug_{backend}.jpg")
                cv2.imwrite(debug_path, debug_img)
                                
                logger.info(f"Face detected using {backend} backend. Cropped face saved to {cropped_path}")
                logger.info(f"Debug image with facial area outline saved to {debug_path}")
                
                return cropped_path
                
            except Exception as e:
                logger.warning(f"Face detection failed with {backend}: {str(e)}")
                continue
                
        raise ValueError("Could not detect face with any backend")

class ImageIndexManager:
    def __init__(self):
        self.index = None
        self.vector_to_image_map = {}
        self.initialize_index()

    def initialize_index(self):
        """Initialize or load existing FAISS index with proper error handling"""
        try:
            if os.path.exists(INDEX_FILE) and os.path.exists(MAPPING_FILE):
                self.load_existing_index()
            else:
                logger.info("Creating new index")
                self.create_new_index()
        except Exception as e:
            logger.error(f"Error during index initialization: {str(e)}")
            logger.info("Falling back to new index creation")
            self.create_new_index()

    def create_new_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatIP(DIMENSION)
        self.vector_to_image_map = {}
        logger.info(f"Created new index with dimension {DIMENSION}")

    def load_existing_index(self):
        """Load existing index with validation"""
        try:
            temp_index = faiss.read_index(INDEX_FILE)
            if temp_index.d != DIMENSION:
                raise ValueError(f"Index dimension mismatch: expected {DIMENSION}, got {temp_index.d}")
            
            with open(MAPPING_FILE, 'rb') as f:
                temp_map = pickle.load(f)
            
            self.index = temp_index
            self.vector_to_image_map = temp_map
            logger.info(f"Successfully loaded existing index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error loading existing index: {str(e)}")
            raise

    def save_index_and_mapping(self):
        """Save index and mapping with proper error handling"""
        try:
            if self.index is None:
                raise ValueError("Cannot save None index")
            
            # Save index
            faiss.write_index(self.index, INDEX_FILE)
            
            # Save mapping
            with open(MAPPING_FILE, 'wb') as f:
                pickle.dump(self.vector_to_image_map, f)
            
            logger.info("Successfully saved index and mapping")
        except Exception as e:
            logger.error(f"Error saving index and mapping: {str(e)}")
            raise

    @staticmethod
    def normalize_vector(v):
        """Normalize vector with validation"""
        norm = np.linalg.norm(v)
        if norm == 0:
            logger.warning("Zero norm vector encountered")
            return v
        return v / norm

class FaceEmbeddingManager:
    def __init__(self, threshold=SIMILARITY_THRESHOLD):
        """Initialize face embedding manager with advanced configurations"""
        self.index = faiss.IndexFlatL2(DIMENSION)  # Use L2 distance for better similarity search
        self.model_name = "Facenet512"
        self.vector_to_image_map = {}
        self.threshold = threshold
        self.face_processor = FaceProcessor()

    def compute_cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two embeddings"""
        return 1 - cosine(vec1, vec2)

    def get_robust_embedding(self, img_path):
        """
        Get face embedding with face detection and alignment
        """
        try:
            # Detect and crop face
            cropped_face_path = self.face_processor.detect_and_crop_face(img_path)
            
            # Get embedding from cropped face using DeepFace
            embeddings = DeepFace.represent(
                img_path=cropped_face_path,
                model_name=self.model_name,
                enforce_detection=False,  # Face already detected
                detector_backend='skip'
            )
            
            # Convert embedding to numpy array
            if isinstance(embeddings, list):
                embedding = np.array(embeddings[0]['embedding'])
            else:
                embedding = np.array(embeddings['embedding'])
                
            embedding = embedding.astype('float32')
            
            # Cleanup
            if os.path.exists(cropped_face_path):
                os.remove(cropped_face_path)
            
            # Normalize embedding
            return embedding / np.linalg.norm(embedding)
            
        except Exception as e:
            logger.error(f"Robust embedding extraction failed: {e}")
            raise

    def add_embedding(self, embedding, image_path):
        try:
            # Validate and normalize embedding
            if embedding.ndim > 1:
                embedding = embedding.squeeze()
            
            embedding = embedding.astype('float32')
            normalized_embedding = embedding / np.linalg.norm(embedding)
            
            # Add to FAISS index
            self.index.add(normalized_embedding.reshape(1, -1))
            new_index = self.index.ntotal - 1
            self.vector_to_image_map[new_index] = image_path
            
            return new_index
        
        except Exception as e:
            logger.error(f"Error adding embedding: {e}")
            raise

    def search_similar_images(self, search_embedding):
        try:
            search_embedding = search_embedding.astype('float32').reshape(1, -1)
            
            if self.index.ntotal == 0:
                logger.warning("No embeddings in the index")
                return None, None, False

            # Perform similarity search
            k = min(self.index.ntotal, 100)  # Search top 100 or all if less
            distances, indices = self.index.search(search_embedding, k)
            
            # Compute cosine similarities
            similarities = []
            for i, idx in enumerate(indices[0]):
                if idx < 0:  # Skip invalid indices
                    continue
                ref_embedding = self.index.reconstruct(int(idx))
                similarity = self.compute_cosine_similarity(search_embedding[0], ref_embedding)
                similarities.append((idx, similarity))
            
            # Sort by similarity in descending order
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Check if best match exceeds threshold
            if similarities and similarities[0][1] > self.threshold:
                best_idx, best_similarity = similarities[0]
                matched_image_path = self.vector_to_image_map[int(best_idx)]
                return matched_image_path, float(best_similarity), True
            
            return None, None, False
        
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            raise

# Initialize FastAPI app and index manager
app = FastAPI()
index_manager = ImageIndexManager()
face_embedding_manager = FaceEmbeddingManager()
s3_helper = S3Helper()

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def save_index_and_mapping():
    # Save index and mapping to temporary location
    temp_index_path = "/tmp/faiss_index.bin"
    temp_mapping_path = "/tmp/vector_to_image_map.pkl"
    faiss.write_index(index, temp_index_path)
    with open(temp_mapping_path, 'wb') as f:
        pickle.dump(vector_to_image_map, f)

    # Upload index and mapping to S3 using S3 helper
    uploaded_index = s3_helper.upload_file(temp_index_path, INDEX_FILE)
    uploaded_mapping = s3_helper.upload_file(temp_mapping_path, MAPPING_FILE)

    # Remove temporary files
    os.remove(temp_index_path)
    os.remove(temp_mapping_path)

    if not uploaded_index or not uploaded_mapping:
        raise ValueError("Failed to upload index or mapping to S3")


def load_index_and_mapping():
    global index, vector_to_image_map
    try:
        # Download index and mapping from S3 using S3 helper
        temp_index_path = "/tmp/faiss_index.bin"
        temp_mapping_path = "/tmp/vector_to_image_map.pkl"
        s3_helper.s3_client.download_file(S3_BUCKET_NAME, INDEX_FILE, temp_index_path)
        s3_helper.s3_client.download_file(S3_BUCKET_NAME, MAPPING_FILE, temp_mapping_path)

        # Load index and mapping
        index = faiss.read_index(temp_index_path)
        with open(temp_mapping_path, 'rb') as f:
            vector_to_image_map = pickle.load(f)
        print(f"Loaded existing index with dimension {index.d}")

        # Remove temporary files
        os.remove(temp_index_path)
        os.remove(temp_mapping_path)
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        vector_to_image_map = {}


def upload_file(self, file_bytes, object_name):
    try:
        self.s3_client.put_object(Body=file_bytes, Bucket=self.s3_bucket, Key=object_name)
    except ClientError as e:
        return False
    return True    

@app.post("/store_image")
async def store_image(image_request: ImageRequest, image_path: str = Body(...)):
    temp_image_path = "/tmp/temp_image.jpg"
    try:
        # Download image
        try:
            req = urllib.request.Request(
                image_request.image_url,
                headers={'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'}
            )
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to download image. Status: {response.status}")
                with open(temp_image_path, 'wb') as f:
                    f.write(response.read())
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error downloading image: {str(e)}")

        # Validate image
        try:
            with Image.open(temp_image_path) as img:
                if not img.format:
                    raise ValueError("Invalid image format")
                logger.info(f"Valid image format: {img.format}")
        except Exception as e:
            raise ValueError(f"Invalid image: {str(e)}")
        
        # Get face embedding with detection and cropping
        embedding = face_embedding_manager.get_robust_embedding(temp_image_path)
        if embedding is None:
            raise ValueError("Failed to generate face embedding")

        # Upload original image to S3
        s3_path = f"{S3_IMAGE_FOLDER}{image_path}"
        if not s3_helper.upload_file(temp_image_path, s3_path):
            raise HTTPException(
                status_code=500,
                detail="Failed to upload image to S3"
            )

        # Add embedding to index
        new_index = face_embedding_manager.add_embedding(embedding, s3_path)
        
        # Save updated index
        index_manager.save_index_and_mapping()

        return {
            "message": "Face image processed and stored successfully",
            "image_path": s3_path,
            "index": new_index
        }

    except ValueError as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

@app.post("/compare_image")
async def compare_image(image_request: ImageRequest):
    temp_image_path = "/tmp/temp_search_image.jpg"
    try:
        # Download image
        try:
            req = urllib.request.Request(
                image_request.image_url,
                headers={'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'}
            )
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to download image. Status: {response.status}")
                with open(temp_image_path, 'wb') as f:
                    f.write(response.read())
        except Exception as e:
            logger.error(f"Failed to download image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error downloading image: {str(e)}")

        # Get face embedding with detection and cropping
        search_embedding = face_embedding_manager.get_robust_embedding(temp_image_path)
        if search_embedding is None:
            raise ValueError("Failed to generate face embedding for the search image")

        # Search for similar faces
        matched_image_path, similarity, is_matched = face_embedding_manager.search_similar_images(search_embedding)

        if not is_matched:
            return {
                "is_matched": False,
                "message": "No matching face found above the threshold"
            }

        # Generate presigned URL for the matched image
        presigned_url = s3_helper.create_presigned_url(matched_image_path)
        if presigned_url is None:
            raise HTTPException(status_code=500, detail="Failed to generate presigned URL")

        logger.info(f"Face match found: {matched_image_path}, similarity: {similarity}")

        return {
            "is_matched": True,
            "matched_image_path": presigned_url,
            "similarity": similarity
        }

    except ValueError as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)     
    
if __name__ == '__main__':
    app.run(host = '0.0.0.0',debug=True)