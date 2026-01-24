"""
Visual Detector Module
Detects signatures and stamps using YOLO and OpenCV fallback.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path


class VisualDetector:
    """
    Detects visual elements (signatures, stamps) in document images.
    Uses YOLO for learning-based detection with OpenCV fallback.
    """
    
    # Class names for custom trained model (5 classes - granular)
    # Match the classes defined in yolo_dataset/data.yaml
    CLASS_NAMES = {
        # Stamps
        0: 'Circular_Stamp',      # Round official seals
        1: 'Rubber_Stamp',        # Rectangular address stamps
        2: 'Ink_Stamp',           # Faded ink marks, misc
        # Signatures
        3: 'Signature',           # Standalone signature
        4: 'Signature_On_Stamp'   # Signature over/inside stamp
    }
    
    # Mapping to output categories (for pipeline compatibility)
    STAMP_CLASSES = {'Circular_Stamp', 'Rubber_Stamp', 'Ink_Stamp'}
    SIGNATURE_CLASSES = {'Signature', 'Signature_On_Stamp'}
    
    def __init__(self, use_yolo: bool = True, confidence_threshold: float = 0.25):
        """
        Initialize visual detector.
        
        Args:
            use_yolo: Whether to use YOLO (requires model weights)
            confidence_threshold: Minimum confidence for detections
        """
        self.use_yolo = use_yolo
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._initialized = False
        self._custom_model = False  # Flag for custom trained model
    
    def _initialize_yolo(self):
        """Initialize YOLO model - prefers ONNX, then .pt, then pretrained."""
        if self._initialized:
            return
        
        try:
            from ultralytics import YOLO
            
            models_dir = Path(__file__).parent.parent / 'models'
            
            # Prefer ONNX for faster inference
            onnx_path = models_dir / 'signature_stamp_yolo.onnx'
            pt_path = models_dir / 'signature_stamp_yolo.pt'
            
            if onnx_path.exists():
                self._model = YOLO(str(onnx_path))
                self._custom_model = True
                print(f"✅ Loaded ONNX model: {onnx_path.name} (faster inference)")
            elif pt_path.exists():
                self._model = YOLO(str(pt_path))
                self._custom_model = True
                print(f"✅ Loaded PyTorch model: {pt_path.name}")
            else:
                # Fallback to YOLO26 pretrained (latest, faster, NMS-free)
                self._model = YOLO('yolo26n.pt')
                self._custom_model = False
                print("⚠️  Using pretrained YOLO26n - custom model not found at models/")
            
            self._initialized = True
            
        except ImportError:
            print("Warning: ultralytics not installed, using OpenCV fallback")
            self.use_yolo = False
        except Exception as e:
            print(f"Warning: YOLO init failed ({e}), using OpenCV fallback")
            self.use_yolo = False
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect signatures and stamps in image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with signature and stamp detections
        """
        results = {
            'signature': {'present': False, 'bbox': None, 'confidence': 0.0},
            'stamp': {'present': False, 'bbox': None, 'confidence': 0.0}
        }
        
        # Try YOLO first (if available and configured)
        if self.use_yolo:
            self._initialize_yolo()
            if self._model:
                yolo_results = self._detect_with_yolo(image)
                results.update(yolo_results)
        
        # OpenCV fallback for stamp detection (circular objects)
        if not results['stamp']['present']:
            stamp_result = self._detect_stamp_opencv(image)
            if stamp_result['present']:
                results['stamp'] = stamp_result
        
        # OpenCV fallback for signature detection
        if not results['signature']['present']:
            sig_result = self._detect_signature_opencv(image)
            if sig_result['present']:
                results['signature'] = sig_result
        
        return results
    
    def _detect_with_yolo(self, image: np.ndarray) -> Dict[str, Any]:
        """Run YOLO detection using custom trained model.
        
        Model classes (5 granular):
          0: Circular_Stamp    -> maps to 'stamp'
          1: Rubber_Stamp      -> maps to 'stamp'
          2: Ink_Stamp         -> maps to 'stamp'
          3: Signature         -> maps to 'signature'
          4: Signature_On_Stamp -> maps to 'signature'
        """
        results = {
            'signature': {'present': False, 'bbox': None, 'confidence': 0.0, 'type': None},
            'stamp': {'present': False, 'bbox': None, 'confidence': 0.0, 'type': None}
        }
        
        # Skip if using pretrained model (doesn't have signature/stamp classes)
        if not self._custom_model:
            return results
        
        # Run inference
        predictions = self._model(image, conf=self.confidence_threshold, verbose=False)
        
        if len(predictions) == 0 or predictions[0].boxes is None:
            return results
        
        boxes = predictions[0].boxes
        
        # Track best detection for each output category
        best_signature = {'conf': 0, 'bbox': None, 'type': None}
        best_stamp = {'conf': 0, 'bbox': None, 'type': None}
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            bbox = [int(x) for x in bbox]  # Convert to integers
            
            class_name = self.CLASS_NAMES.get(cls_id, '')
            
            # Map 5 classes to 2 output categories
            if class_name in self.SIGNATURE_CLASSES and conf > best_signature['conf']:
                best_signature = {'conf': conf, 'bbox': bbox, 'type': class_name}
            elif class_name in self.STAMP_CLASSES and conf > best_stamp['conf']:
                best_stamp = {'conf': conf, 'bbox': bbox, 'type': class_name}
        
        # Update results with best detections
        if best_signature['bbox']:
            results['signature'] = {
                'present': True,
                'bbox': best_signature['bbox'],
                'confidence': best_signature['conf'],
                'type': best_signature['type']
            }
        
        if best_stamp['bbox']:
            results['stamp'] = {
                'present': True,
                'bbox': best_stamp['bbox'],
                'confidence': best_stamp['conf'],
                'type': best_stamp['type']
            }
        
        return results
    
    def _detect_stamp_opencv(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect circular stamps using OpenCV.
        
        Stamps are typically:
        - Circular or oval shaped
        - Located in bottom portion of document (left, center, or right)
        - Have distinct colored edges (blue, red, purple)
        """
        result = {'present': False, 'bbox': None, 'confidence': 0.0}
        
        h, w = image.shape[:2]
        
        # Search ENTIRE bottom 40% of the document (stamps can be anywhere)
        roi_y_start = int(h * 0.6)
        roi_x_start = 0  # Search entire width
        roi = image[roi_y_start:, roi_x_start:]
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough Transform with stricter parameters
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=60,  # Higher threshold for edge detection
            param2=35,  # Higher threshold for circle detection
            minRadius=40,  # Minimum stamp radius
            maxRadius=120  # Maximum stamp radius
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            best_circle = None
            best_score = 0
            
            for circle in circles[0]:
                x, y, r = circle
                
                # Check if it's in a good stamp location
                score = r  # Base score on size
                
                # Prefer circles in the bottom-right corner
                if x > roi.shape[1] * 0.3:  # Right side of ROI
                    score *= 1.5
                if y > roi.shape[0] * 0.3:  # Bottom of ROI
                    score *= 1.3
                
                # Additional check: stamps often have colored edges
                # Check if the circle area has color variation (not just text)
                if len(roi.shape) == 3:
                    cx, cy = int(x), int(y)
                    r_int = int(r)
                    if 0 <= cy-r_int < roi.shape[0] and 0 <= cx-r_int < roi.shape[1]:
                        circle_region = roi[max(0,cy-r_int):min(roi.shape[0],cy+r_int),
                                           max(0,cx-r_int):min(roi.shape[1],cx+r_int)]
                        if circle_region.size > 0:
                            # Check for color variation (stamps are usually colored)
                            hsv = cv2.cvtColor(circle_region, cv2.COLOR_BGR2HSV)
                            saturation = hsv[:,:,1].mean()
                            if saturation > 20:  # Has some color
                                score *= 1.5
                
                if score > best_score:
                    best_score = score
                    best_circle = circle
            
            if best_circle is not None:
                x, y, r = best_circle
                # Convert to original image coordinates
                abs_x = roi_x_start + int(x)
                abs_y = roi_y_start + int(y)
                
                # Create bounding box
                bbox = [
                    max(0, abs_x - int(r)),
                    max(0, abs_y - int(r)),
                    min(w, abs_x + int(r)),
                    min(h, abs_y + int(r))
                ]
                
                result = {
                    'present': True,
                    'bbox': bbox,
                    'confidence': min(0.85, 0.4 + (best_score / 300))
                }
        
        # Also try contour-based detection for non-circular stamps
        if not result['present']:
            result = self._detect_stamp_contour(image)
        
        return result
    
    def _detect_stamp_contour(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect stamps using contour analysis."""
        result = {'present': False, 'bbox': None, 'confidence': 0.0}
        
        h, w = image.shape[:2]
        
        # Focus on bottom half
        roi = image[int(h * 0.5):, :]
        
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (stamps are typically 2000-50000 pixels)
            if 2000 < area < 50000:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    # Stamps tend to be circular (0.7-1.0)
                    if circularity > 0.5:
                        x, y, bw, bh = cv2.boundingRect(contour)
                        
                        # Adjust y coordinate for ROI offset
                        abs_y = int(h * 0.5) + y
                        
                        result = {
                            'present': True,
                            'bbox': [x, abs_y, x + bw, abs_y + bh],
                            'confidence': min(0.8, circularity)
                        }
                        break
        
        return result
    
    def _detect_signature_opencv(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect handwritten signatures using multiple strategies.
        
        Strategies:
        1. Contour analysis for ink strokes
        2. Template matching for "Signature" text area
        3. Stroke density analysis
        """
        result = {'present': False, 'bbox': None, 'confidence': 0.0}
        
        h, w = image.shape[:2]
        
        # Strategy 1: Contour-based detection in bottom portion
        # Focus on bottom 40% (signatures usually at bottom)
        roi_y_start = int(h * 0.6)
        roi = image[roi_y_start:, :]
        
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Threshold to isolate dark ink
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        signature_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # RELAXED: Signatures can vary in size (300-30000 pixels)
            if 300 < area < 30000:
                x, y, bw, bh = cv2.boundingRect(contour)
                
                # RELAXED: Accept various aspect ratios
                aspect_ratio = bw / (bh + 1)
                
                # Accept wider range of aspect ratios (1.0 to 15)
                if 1.0 < aspect_ratio < 15 and bw > 40:
                    # Check stroke characteristics
                    perimeter = cv2.arcLength(contour, True)
                    complexity = perimeter ** 2 / (area + 1)
                    
                    # RELAXED: Lower complexity threshold
                    if complexity > 15:
                        abs_y = roi_y_start + y
                        signature_candidates.append({
                            'bbox': [x, abs_y, x + bw, abs_y + bh],
                            'score': complexity * (bw / 100),
                            'method': 'contour'
                        })
        
        # Strategy 2: Look for signature-like regions near "Signature" text
        # If no candidates found, try alternate approach
        if not signature_candidates:
            # Try with lower ROI (bottom 30%)
            alt_roi_start = int(h * 0.7)
            alt_roi = image[alt_roi_start:, :]
            
            if len(alt_roi.shape) == 3:
                alt_gray = cv2.cvtColor(alt_roi, cv2.COLOR_BGR2GRAY)
            else:
                alt_gray = alt_roi.copy()
            
            # Use adaptive thresholding for better ink detection
            alt_binary = cv2.adaptiveThreshold(
                alt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find stroke-like regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            dilated = cv2.dilate(alt_binary, kernel, iterations=1)
            
            alt_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in alt_contours:
                area = cv2.contourArea(contour)
                if 200 < area < 50000:
                    x, y, bw, bh = cv2.boundingRect(contour)
                    aspect_ratio = bw / (bh + 1)
                    
                    # Look for horizontal stroke patterns
                    if aspect_ratio > 1.2 and bw > 30 and bh < 100:
                        abs_y = alt_roi_start + y
                        signature_candidates.append({
                            'bbox': [x, abs_y, x + bw, abs_y + bh],
                            'score': (bw * bh) / 1000,
                            'method': 'adaptive'
                        })
        
        # Strategy 3: Cluster nearby candidates into single signature
        if signature_candidates:
            # Merge overlapping candidates
            merged = self._merge_nearby_boxes(signature_candidates)
            
            # Choose best candidate
            best = max(merged, key=lambda x: x['score'])
            result = {
                'present': True,
                'bbox': best['bbox'],
                'confidence': min(0.80, 0.4 + best['score'] / 150)
            }
        
        return result
    
    def _merge_nearby_boxes(self, candidates: List[Dict]) -> List[Dict]:
        """Merge overlapping bounding boxes."""
        if len(candidates) <= 1:
            return candidates
        
        # Sort by x coordinate
        sorted_cands = sorted(candidates, key=lambda x: x['bbox'][0])
        merged = []
        current = sorted_cands[0].copy()
        
        for cand in sorted_cands[1:]:
            # Check if boxes overlap or are close
            x1, y1, x2, y2 = current['bbox']
            nx1, ny1, nx2, ny2 = cand['bbox']
            
            # If close enough horizontally and vertically similar
            if nx1 <= x2 + 50 and abs(ny1 - y1) < 50:
                # Merge boxes
                current['bbox'] = [
                    min(x1, nx1),
                    min(y1, ny1),
                    max(x2, nx2),
                    max(y2, ny2)
                ]
                current['score'] = max(current['score'], cand['score'])
            else:
                merged.append(current)
                current = cand.copy()
        
        merged.append(current)
        return merged
    
    def visualize_detections(self, image: np.ndarray, 
                             detections: Dict[str, Any]) -> np.ndarray:
        """
        Draw detection boxes on image for visualization.
        
        Args:
            image: Original image
            detections: Detection results
            
        Returns:
            Image with drawn boxes
        """
        img_viz = image.copy()
        
        # Draw signature box (green)
        if detections.get('signature', {}).get('present'):
            bbox = detections['signature']['bbox']
            if bbox:
                cv2.rectangle(img_viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)
                cv2.putText(img_viz, 'Signature', (bbox[0], bbox[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw stamp box (blue)
        if detections.get('stamp', {}).get('present'):
            bbox = detections['stamp']['bbox']
            if bbox:
                cv2.rectangle(img_viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                            (255, 0, 0), 2)
                cv2.putText(img_viz, 'Stamp', (bbox[0], bbox[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return img_viz
