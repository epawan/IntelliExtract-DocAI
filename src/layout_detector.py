"""
Layout Detection Module for Document AI
Uses OpenCV to detect document regions (header, table, footer) before OCR.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectedRegion:
    """Container for detected document region."""
    region_type: str  # 'header', 'table', 'footer', 'text_block'
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    position: str  # 'top-left', 'top-right', 'center', etc.
    estimated_font_size: str  # 'large', 'medium', 'small'
    area: int


class LayoutDetector:
    """
    Detects document layout structure using OpenCV.
    
    Pipeline:
    1. Preprocess image (grayscale, threshold, morphology)
    2. Detect text regions using contours
    3. Classify regions by position and size
    4. Detect table structures
    """
    
    def __init__(self):
        """Initialize layout detector."""
        self.min_text_area = 2000  # Minimum area for text region (increased)
        self.max_text_area = 500000  # Maximum area for text region
        self.min_aspect_ratio = 0.1  # Minimum width/height ratio
        self.max_aspect_ratio = 20.0  # Maximum width/height ratio
        
    def preprocess_for_layout(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for layout detection.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Morphological operations to connect text into blocks
        # Horizontal kernel to connect characters into words
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 2))
        h_dilated = cv2.dilate(binary, h_kernel, iterations=2)
        
        # Vertical kernel to connect lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
        dilated = cv2.dilate(h_dilated, v_kernel, iterations=1)
        
        return dilated
    
    def detect_text_regions(self, image: np.ndarray) -> List[DetectedRegion]:
        """
        Detect all text regions in the image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of detected regions
        """
        h, w = image.shape[:2]
        
        # Preprocess
        processed = self.preprocess_for_layout(image)
        
        # Find contours
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        
        for contour in contours:
            # Get bounding box
            x, y, w_box, h_box = cv2.boundingRect(contour)
            area = w_box * h_box
            
            # Filter by area
            if area < self.min_text_area or area > self.max_text_area:
                continue
            
            # Filter by aspect ratio (remove very thin or very wide regions)
            aspect_ratio = w_box / h_box if h_box > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            
            # Classify region
            region_type = self._classify_region_type(x, y, w_box, h_box, w, h)
            position = self._get_position(x, y, w_box, h_box, w, h)
            font_size = self._estimate_font_size(w_box, h_box, area)
            
            # Calculate confidence based on region characteristics
            confidence = self._calculate_region_confidence(
                region_type, position, font_size, area
            )
            
            regions.append(DetectedRegion(
                region_type=region_type,
                bbox=(x, y, w_box, h_box),
                confidence=confidence,
                position=position,
                estimated_font_size=font_size,
                area=area
            ))
        
        # Sort by y-position (top to bottom)
        regions.sort(key=lambda r: r.bbox[1])
        
        return regions
    
    def _classify_region_type(self, x: int, y: int, w: int, h: int, 
                              img_w: int, img_h: int) -> str:
        """
        Classify region type based on position and size.
        
        Args:
            x, y, w, h: Bounding box
            img_w, img_h: Image dimensions
            
        Returns:
            Region type: 'header', 'table', 'footer', 'text_block'
        """
        y_ratio = y / img_h
        
        # Header region (top 25%)
        if y_ratio < 0.25:
            return 'header'
        
        # Footer region (bottom 20%)
        elif y_ratio > 0.80:
            return 'footer'
        
        # Table region (middle, wide aspect ratio)
        elif 0.25 <= y_ratio <= 0.80:
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 3:  # Wide horizontal region
                return 'table'
            else:
                return 'text_block'
        
        return 'text_block'
    
    def _get_position(self, x: int, y: int, w: int, h: int,
                     img_w: int, img_h: int) -> str:
        """
        Get position descriptor for region.
        
        Returns:
            Position string like 'top-left', 'top-right', 'center', etc.
        """
        x_center = x + w / 2
        y_center = y + h / 2
        
        # Vertical position
        if y_center < img_h * 0.33:
            v_pos = 'top'
        elif y_center < img_h * 0.67:
            v_pos = 'middle'
        else:
            v_pos = 'bottom'
        
        # Horizontal position
        if x_center < img_w * 0.33:
            h_pos = 'left'
        elif x_center < img_w * 0.67:
            h_pos = 'center'
        else:
            h_pos = 'right'
        
        return f"{v_pos}-{h_pos}"
    
    def _estimate_font_size(self, w: int, h: int, area: int) -> str:
        """
        Estimate font size based on bounding box dimensions.
        
        Returns:
            'large', 'medium', or 'small'
        """
        # Use height as primary indicator
        if h > 60:
            return 'large'
        elif h > 30:
            return 'medium'
        else:
            return 'small'
    
    def _calculate_region_confidence(self, region_type: str, position: str,
                                     font_size: str, area: int) -> float:
        """
        Calculate confidence score for detected region.
        
        Returns:
            Confidence score 0.0-1.0
        """
        confidence = 0.5  # Base confidence
        
        # Boost for header regions in top-right (likely dealer name)
        if region_type == 'header' and 'right' in position:
            confidence += 0.2
        
        # Boost for large font in header
        if region_type == 'header' and font_size == 'large':
            confidence += 0.15
        
        # Boost for table regions
        if region_type == 'table':
            confidence += 0.1
        
        # Boost for reasonable area
        if 1000 < area < 100000:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def detect_table_structure(self, image: np.ndarray, 
                               region_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Detect table structure within a region.
        
        Args:
            image: Input image
            region_bbox: (x, y, w, h) of table region
            
        Returns:
            Dict with table structure info (rows, columns, cells)
        """
        x, y, w, h = region_bbox
        
        # Crop to table region
        table_img = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        if len(table_img.shape) == 3:
            gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = table_img.copy()
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find line positions
        h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract row/column positions
        row_positions = sorted([cv2.boundingRect(c)[1] for c in h_contours])
        col_positions = sorted([cv2.boundingRect(c)[0] for c in v_contours])
        
        return {
            'has_table': len(row_positions) > 1 or len(col_positions) > 1,
            'num_rows': len(row_positions) - 1 if len(row_positions) > 1 else 0,
            'num_cols': len(col_positions) - 1 if len(col_positions) > 1 else 0,
            'row_positions': row_positions,
            'col_positions': col_positions
        }
    
    def find_dealer_name_region(self, regions: List[DetectedRegion]) -> Optional[DetectedRegion]:
        """
        Find the most likely dealer name region.
        
        Dealer names are typically:
        - In header (top 25%)
        - Top-right position (letterhead)
        - Large font
        
        Args:
            regions: List of detected regions
            
        Returns:
            Best candidate region for dealer name
        """
        candidates = []
        
        for region in regions:
            if region.region_type != 'header':
                continue
            
            score = 0
            
            # Prefer top-right position
            if 'right' in region.position:
                score += 50
            
            # Prefer large font
            if region.estimated_font_size == 'large':
                score += 30
            
            # Prefer top area
            if 'top' in region.position:
                score += 20
            
            if score > 0:
                candidates.append((region, score))
        
        if candidates:
            candidates.sort(key=lambda x: -x[1])
            return candidates[0][0]
        
        return None
    
    def find_table_regions(self, regions: List[DetectedRegion]) -> List[DetectedRegion]:
        """
        Find all table regions.
        
        Args:
            regions: List of detected regions
            
        Returns:
            List of table regions
        """
        return [r for r in regions if r.region_type == 'table']


# Export
__all__ = ['LayoutDetector', 'DetectedRegion']
