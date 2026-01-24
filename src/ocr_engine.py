"""
OCR Engine Module
Handles text extraction from images using EasyOCR (primary) and PaddleOCR (fallback).
Supports multiple languages including English, Hindi, and Gujarati.
EasyOCR is preferred for CPU stability. PaddleOCR used with OneDNN disabled.
"""

# CRITICAL: Disable OneDNN/MKL before importing Paddle to avoid CPU inference bug
import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_oneDNN"] = "0"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
# Fix for std::exception / OpenMP crashes on CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import List, Dict, Tuple, Any, Optional
import numpy as np


class OCRResult:
    """Container for OCR results with text, bounding box, and confidence."""
    
    def __init__(self, text: str, bbox: List[List[int]], confidence: float):
        self.text = text
        self.bbox = bbox  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        self.confidence = confidence
    
    @property
    def x1(self) -> int:
        return min(p[0] for p in self.bbox)
    
    @property
    def y1(self) -> int:
        return min(p[1] for p in self.bbox)
    
    @property
    def x2(self) -> int:
        return max(p[0] for p in self.bbox)
    
    @property
    def y2(self) -> int:
        return max(p[1] for p in self.bbox)
    
    @property
    def rect_bbox(self) -> List[int]:
        """Return as [x1, y1, x2, y2] format."""
        return [self.x1, self.y1, self.x2, self.y2]
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return center point of bbox."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def __repr__(self):
        return f"OCRResult('{self.text[:30]}...', conf={self.confidence:.2f})"


class OCREngine:
    """
    OCR Engine using PaddleOCR (v2.7.x) for multilingual text extraction.
    Supports Hindi, Gujarati, and English text recognition.
    """
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = False, fast_mode: bool = False):
        """
        Initialize OCR Engine.
        
        Args:
            languages: List of language codes (e.g., ['en', 'hi'])
            use_gpu: Whether to use GPU acceleration
            fast_mode: If True, use only Tesseract (fast), skip ensemble (20x faster)
        """
        self.languages = languages or ['en']
        self.use_gpu = use_gpu
        self.fast_mode = fast_mode
        self._ocr = None
        self._ocr_engines = {}
        self._initialized = False
        self._engine_type = None
        self._use_ensemble = False
    
    def _initialize(self):
        """Lazy initialization of OCR engine (PaddleOCR + EasyOCR fallback)."""
        if self._initialized:
            return
        
        # Initialize all available OCR engines
        self._ocr_engines = {}
        
        # 1. Initialize PaddleOCR PP-OCRv5 (primary engine)
        try:
            from paddleocr import PaddleOCR
            print(f"  Initializing PaddleOCR PP-OCRv5 (Devanagari/Hindi) with enhanced settings...")
            self._ocr_engines['paddle'] = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="devanagari_PP-OCRv5_mobile_rec",
                use_angle_cls=True,               # better handling of rotated headings
                det_db_thresh=0.3,                # lower threshold to capture large/faint text
            )
            print(f"  ✅ PaddleOCR PP-OCRv5 initialized with angle classification and lower detection threshold")
        except Exception as e:
            print(f"  ⚠️ PaddleOCR not available: {e}")
        
        # 2. Try EasyOCR (fallback/ensemble)
        if not self.fast_mode:
            try:
                import easyocr
                supported_langs = {'en', 'hi', 'mr', 'ta', 'te', 'bn', 'pa'}
                ocr_langs = [l for l in self.languages if l in supported_langs]
                if not ocr_langs:
                    ocr_langs = ['en']
                
                print(f"  Initializing EasyOCR fallback with languages: {ocr_langs}")
                # Use CPU for EasyOCR stability if use_gpu=False
                self._ocr_engines['easyocr'] = easyocr.Reader(ocr_langs, gpu=self.use_gpu, verbose=False)
                print(f"  ✅ EasyOCR initialized successfully")
            except Exception as e:
                print(f"  ⚠️ EasyOCR initialization failed: {e}")
        
        if not self._ocr_engines:
            # Re-check if EasyOCR can be used as absolute fallback if Paddle fails
            try:
                import easyocr
                self._ocr_engines['easyocr'] = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)
            except:
                raise ImportError("No OCR engine could be initialized (PaddleOCR and EasyOCR failed)")
        
        # Set primary engine (prefer PaddleOCR)
        if 'paddle' in self._ocr_engines:
            self._engine_type = 'paddle'
            self._ocr = self._ocr_engines['paddle']
        elif 'easyocr' in self._ocr_engines:
            self._engine_type = 'easyocr'
            self._ocr = self._ocr_engines['easyocr']
        
        # Disable ensemble mode (user request: use only single OCR)
        self._use_ensemble = False
        if 'paddle' in self._ocr_engines:
            self._engine_type = 'paddle'
            self._ocr = self._ocr_engines['paddle']
        elif 'easyocr' in self._ocr_engines:
            self._engine_type = 'easyocr'
            self._ocr = self._ocr_engines['easyocr']
        
        self._initialized = True
    
    def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        """
        Extract text from image using ensemble OCR if available.
        
        Args:
            image: Image as numpy array (BGR or grayscale)
            
        Returns:
            List of OCRResult objects with text, bbox, and confidence
        """
        self._initialize()
        
        results = []
        
        try:
            if self._engine_type == 'paddle':
                try:
                    results = self._extract_with_paddleocr(image)
                except Exception as e:
                    print(f"  ⚠️ PaddleOCR runtime crash: {e}")
                    if 'easyocr' in self._ocr_engines:
                        print("  🔄 Falling back to EasyOCR stability layer...")
                        results = self._extract_with_easyocr(image)
                    else:
                        raise e
            elif self._engine_type == 'tesseract':
                results = self._extract_with_tesseract(image)
            else:
                results = self._extract_with_easyocr(image)
                
        except Exception as e:
            print(f"  ❌ OCR Error: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    def _extract_with_tesseract(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using Tesseract OCR."""
        results = []
        
        try:
            from PIL import Image
            
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                # BGR to RGB
                import cv2
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = Image.fromarray(image)
            
            # Get detailed OCR data with bounding boxes
            data = self._ocr.image_to_data(pil_image, output_type=self._ocr.Output.DICT)
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if not text:  # Skip empty text
                    continue
                
                conf = int(data['conf'][i])
                if conf < 0:  # Tesseract returns -1 for invalid
                    continue
                
                # Get bounding box coordinates
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                # Convert to our bbox format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                bbox = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]
                
                # Tesseract confidence is 0-100, normalize to 0-1
                confidence = conf / 100.0
                
                results.append(OCRResult(text, bbox, confidence))
        
        except Exception as e:
            print(f"  Tesseract extraction error: {e}")
        
        return results
    
    def _extract_with_paddleocr(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using PaddleOCR PP-OCRv5 predict() API."""
        results = []
        
        # PP-OCRv5 uses predict() method
        ocr_output = self._ocr.predict(image)
        
        if ocr_output is None:
            return results
        
        # Parse PP-OCRv5 output format: list of pages with rec_texts, rec_scores, rec_boxes
        for page in ocr_output:
            if page is None or not isinstance(page, dict):
                continue
            
            rec_texts = page.get("rec_texts", [])
            rec_scores = page.get("rec_scores", [])
            rec_boxes = page.get("rec_boxes", [])
            
            for i, text in enumerate(rec_texts):
                if not text:
                    continue
                
                score = float(rec_scores[i]) if i < len(rec_scores) else 0.8
                
                # Parse bounding box
                if i < len(rec_boxes):
                    box = rec_boxes[i]
                    # rec_boxes format: [x1, y1, x2, y2] or polygon
                    if len(box) == 4 and not isinstance(box[0], (list, tuple)):
                        # [x1, y1, x2, y2] format
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    else:
                        # Polygon format
                        bbox = [[int(p[0]), int(p[1])] for p in box]
                else:
                    bbox = [[0, 0], [100, 0], [100, 30], [0, 30]]
                
                results.append(OCRResult(str(text), bbox, score))
        
        return results
    
    def _parse_paddleocr_output(self, ocr_output, results: List[OCRResult]):
        """Parse PaddleOCR output from various versions."""
        if isinstance(ocr_output, dict):
            # New format with rec_text, dt_polys etc
            rec_texts = ocr_output.get('rec_text', [])
            rec_scores = ocr_output.get('rec_score', [])
            dt_polys = ocr_output.get('dt_polys', [])
            
            for i, text in enumerate(rec_texts):
                if text:
                    score = rec_scores[i] if i < len(rec_scores) else 0.8
                    if i < len(dt_polys):
                        poly = dt_polys[i]
                        bbox = [[int(p[0]), int(p[1])] for p in poly]
                    else:
                        bbox = [[0,0], [100,0], [100,30], [0,30]]
                    results.append(OCRResult(str(text), bbox, float(score)))
                    
        elif isinstance(ocr_output, list):
            # Old format: list of pages with [bbox, (text, score)] items
            for page in ocr_output:
                if page is None:
                    continue
                if isinstance(page, dict):
                    self._parse_paddleocr_output(page, results)
                    continue
                for line in page:
                    if line is None or len(line) < 2:
                        continue
                    bbox = line[0]
                    text_info = line[1]
                    if isinstance(text_info, tuple) and len(text_info) >= 2:
                        text = str(text_info[0])
                        confidence = float(text_info[1])
                    else:
                        text = str(text_info)
                        confidence = 0.8
                    bbox_int = [[int(p[0]), int(p[1])] for p in bbox]
                    results.append(OCRResult(text, bbox_int, confidence))
    
    def _extract_with_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using EasyOCR."""
        results = []
        
        # EasyOCR expects RGB, convert if BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            import cv2
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Run EasyOCR
        ocr_output = self._ocr.readtext(image_rgb)
        
        for detection in ocr_output:
            bbox, text, confidence = detection
            bbox_int = [[int(p[0]), int(p[1])] for p in bbox]
            results.append(OCRResult(str(text), bbox_int, float(confidence)))
        
        return results
    
    def extract_text_simple(self, image: np.ndarray) -> str:
        """
        Extract all text as a single string.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Concatenated text string
        """
        results = self.extract_text(image)
        return ' '.join([r.text for r in results])
    
    def get_text_with_positions(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Get structured text with position information.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary with text blocks organized by position
        """
        results = self.extract_text(image)
        
        if not results:
            return {'blocks': [], 'full_text': '', 'raw_results': []}
        
        # Sort by vertical position (top to bottom)
        results_sorted = sorted(results, key=lambda r: r.y1)
        
        # Group into lines based on y-coordinate proximity
        lines = []
        current_line = []
        current_y = -1
        line_threshold = 20  # pixels
        
        for r in results_sorted:
            if current_y < 0 or abs(r.y1 - current_y) < line_threshold:
                current_line.append(r)
                current_y = r.y1
            else:
                if current_line:
                    current_line.sort(key=lambda x: x.x1)
                    lines.append(current_line)
                current_line = [r]
                current_y = r.y1
        
        if current_line:
            current_line.sort(key=lambda x: x.x1)
            lines.append(current_line)
        
        # Build output
        blocks = []
        for line in lines:
            line_text = ' '.join([r.text for r in line])
            avg_y = sum(r.y1 for r in line) // len(line)
            blocks.append({
                'text': line_text,
                'y_position': avg_y,
                'elements': [{'text': r.text, 'bbox': r.rect_bbox, 'conf': r.confidence} for r in line]
            })
        
        full_text = '\n'.join([b['text'] for b in blocks])
        
        return {
            'blocks': blocks,
            'full_text': full_text,
            'raw_results': results
        }
