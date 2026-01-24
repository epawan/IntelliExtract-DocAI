"""
Image Preprocessor Module
Handles PDF to image conversion and image preprocessing for OCR/detection.
"""

import os
from pathlib import Path
from typing import List, Union, Tuple
import numpy as np
from PIL import Image
import cv2


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy array of the image in BGR format (for OpenCV compatibility)
    """
    img = cv2.imread(image_path)
    if img is None:
        # Try with PIL for more format support
        pil_img = Image.open(image_path)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion
        
    Returns:
        List of numpy arrays (images)
    """
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=dpi)
        images = []
        for page in pages:
            img_array = np.array(page)
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            images.append(img_bgr)
        return images
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF: {e}")


def preprocess_for_ocr(image: np.ndarray, aggressive: bool = True) -> np.ndarray:
    """
    Preprocess image for better OCR results with enhanced techniques.
    
    Args:
        image: Input image in BGR format
        aggressive: If True, apply more aggressive preprocessing
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    if aggressive:
        # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Adaptive thresholding for binary image (better for text)
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    else:
        # For most cases, just return denoised grayscale
        return denoised


def resize_image(image: np.ndarray, max_size: int = 2000) -> Tuple[np.ndarray, float]:
    """
    Resize image if too large while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Tuple of (resized image, scale factor)
    """
    h, w = image.shape[:2]
    scale = 1.0
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image, scale


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Deskew a slightly rotated document image.
    
    Args:
        image: Input image
        
    Returns:
        Deskewed image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Find edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return image
    
    # Calculate average angle
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines
        if abs(angle) < 45:
            angles.append(angle)
    
    if not angles:
        return image
    
    median_angle = np.median(angles)
    
    # Only deskew if angle is significant
    if abs(median_angle) < 0.5:
        return image
    
    # Rotate image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def load_document(file_path: str) -> List[np.ndarray]:
    """
    Load a document (PDF or image) and return list of page images.
    
    Args:
        file_path: Path to the document
        
    Returns:
        List of images (one per page)
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext == '.pdf':
        return pdf_to_images(file_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']:
        img = load_image(file_path)
        return [img]
    else:
        raise ValueError(f"Unsupported file format: {ext}")
