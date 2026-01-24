"""
Confidence Scoring Module
Calculates and calibrates confidence scores for extractions.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ConfidenceWeights:
    """Weights for different confidence factors."""
    ocr_confidence: float = 0.3
    extraction_confidence: float = 0.4
    validation_score: float = 0.3


class ConfidenceScorer:
    """
    Calculates confidence scores for extracted fields.
    """
    
    def __init__(self, weights: ConfidenceWeights = None):
        self.weights = weights or ConfidenceWeights()
    
    def score_field(self, 
                    field_value: Any,
                    ocr_confidence: float,
                    extraction_confidence: float,
                    field_type: str,
                    field_name: str) -> float:
        """
        Calculate confidence score for a single field.
        
        Args:
            field_value: Extracted value
            ocr_confidence: Confidence from OCR
            extraction_confidence: Confidence from extraction method
            field_type: Type of field (text, numeric, currency, visual)
            field_name: Name of the field
            
        Returns:
            Combined confidence score (0.0 to 1.0)
        """
        # Base score from OCR and extraction
        base_score = (
            ocr_confidence * self.weights.ocr_confidence +
            extraction_confidence * self.weights.extraction_confidence
        )
        
        # Validation score based on field type
        validation_score = self._validate_field(field_value, field_type, field_name)
        
        # Combined score
        final_score = base_score + validation_score * self.weights.validation_score
        
        return min(1.0, max(0.0, final_score))
    
    def _validate_field(self, value: Any, field_type: str, field_name: str) -> float:
        """Validate field value and return validation score."""
        if value is None:
            return 0.0
        
        if field_type == 'numeric':
            return self._validate_numeric(value, field_name)
        elif field_type == 'currency':
            return self._validate_currency(value, field_name)
        elif field_type == 'text':
            return self._validate_text(value, field_name)
        elif field_type == 'visual':
            return 0.8 if value else 0.0
        
        return 0.5
    
    def _validate_numeric(self, value: Any, field_name: str) -> float:
        """Validate numeric field."""
        try:
            num = float(value)
            
            if field_name == 'horse_power':
                # Valid HP range: 10-200
                if 10 <= num <= 200:
                    return 1.0
                elif 5 <= num <= 250:
                    return 0.7
                else:
                    return 0.3
            
            return 0.8
        except (ValueError, TypeError):
            return 0.0
    
    def _validate_currency(self, value: Any, field_name: str) -> float:
        """Validate currency field."""
        try:
            amount = float(value)
            
            if field_name == 'asset_cost':
                # Valid tractor cost range: 1 lakh to 50 lakhs
                if 100000 <= amount <= 5000000:
                    # Higher score for rounded numbers
                    if amount % 100 == 0:
                        return 1.0
                    return 0.6  # Penalty for suspiciously precise numbers
                elif 50000 <= amount <= 10000000:
                    return 0.7
                else:
                    return 0.3
            
            return 0.8
        except (ValueError, TypeError):
            return 0.0
    
    def _validate_text(self, value: Any, field_name: str) -> float:
        """Validate text field."""
        if not value or not isinstance(value, str):
            return 0.0
        
        text = str(value).strip()
        
        if len(text) < 3:
            return 0.3
        elif len(text) > 200:
            return 0.5
        
        # Check for gibberish (too many special chars)
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.5:
            return 0.5
        
        return 0.9
    
    def calculate_document_confidence(self, 
                                       field_confidences: Dict[str, float],
                                       required_fields: List[str] = None) -> float:
        """
        Calculate overall document confidence.
        
        Args:
            field_confidences: Dict of field_name -> confidence
            required_fields: List of fields that must be present
            
        Returns:
            Document-level confidence score
        """
        if not field_confidences:
            return 0.0
        
        required_fields = required_fields or [
            'dealer_name', 'model_name', 'horse_power', 
            'asset_cost', 'signature', 'stamp'
        ]
        
        # Check how many required fields are present
        present_fields = [f for f in required_fields if f in field_confidences]
        coverage = len(present_fields) / len(required_fields)
        
        # Average confidence of present fields
        if present_fields:
            avg_confidence = sum(field_confidences[f] for f in present_fields) / len(present_fields)
        else:
            avg_confidence = 0.0
        
        # Document confidence = coverage * average confidence
        doc_confidence = coverage * avg_confidence
        
        return round(doc_confidence, 3)
