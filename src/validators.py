"""
Field Validators for Document AI
Business rule validation for extracted fields.
"""

import re
from typing import Dict, Optional, Tuple


class FieldValidator:
    """Validates extracted field values using business rules."""
    
    def __init__(self):
        """Initialize validator with patterns and rules."""
        # Known tractor brands (should NOT be dealer names)
        self.known_brands = {
            'sonalika', 'mahindra', 'john deere', 'kubota', 'eicher',
            'swaraj', 'new holland', 'massey ferguson', 'tafe', 'escorts',
            'farmtrac', 'powertrac', 'captain', 'indo farm',
            'सोनालीका', 'महिंद्रा', 'स्वराज', 'फार्मट्रैक'
        }
        
        # Form field labels (should NOT be dealer names)
        self.form_labels = {
            'son/d/w', 'name:', 'address:', 'father', 'mother',
            'quotation', 'invoice', 'gstin', 'mobile', 'phone',
            'email', 'date:', 'bill', 'receipt', 'challan',
            'नाम:', 'पता:', 'पिता', 'माता', 'बिल', 'रसीद'
        }
        
        # Company suffixes (SHOULD be in dealer names)
        self.company_suffixes = {
            'tractors', 'motors', 'agency', 'enterprises', 'pvt', 'ltd',
            'company', 'corporation', 'traders', 'sales', 'services',
            'ट्रैक्टर्स', 'मोटर्स', 'एजेंसी', 'एंटरप्राइजेज', 'कंपनी'
        }
    
    def validate_dealer_name(self, text: str) -> Tuple[bool, float, str]:
        """
        Validate dealer name.
        
        Args:
            text: Extracted dealer name
            
        Returns:
            (is_valid, confidence_adjustment, reason)
        """
        if not text or len(text) < 3:
            return False, -0.5, "Too short"
        
        if len(text) > 100:
            return False, -0.3, "Too long"
        
        text_lower = text.lower()
        
        # Check if it's a known brand (bad)
        for brand in self.known_brands:
            if brand in text_lower or brand in text:
                return False, -0.4, f"Known brand: {brand}"
        
        # Check if it's primarily numeric (likely a price or ID)
        digits = sum(c.isdigit() for c in text)
        if digits > len(text) * 0.5:
            return False, -0.6, "Primarily numeric"
            
        # Check if it's a form label (bad)
        for label in self.form_labels:
            if label in text_lower or label in text:
                return False, -0.5, f"Form label: {label}"
        
        # Check for company suffix (good)
        has_suffix = False
        for suffix in self.company_suffixes:
            if suffix in text_lower or suffix in text:
                has_suffix = True
                break
        
        if has_suffix:
            return True, +0.2, "Has company suffix"
        else:
            return True, 0.0, "No company suffix but acceptable"
    
    def validate_model_name(self, text: str) -> Tuple[bool, float, str]:
        """
        Validate model name.
        
        Args:
            text: Extracted model name
            
        Returns:
            (is_valid, confidence_adjustment, reason)
        """
        if not text or len(text) < 3:
            return False, -0.5, "Too short"
        
        if len(text) > 80:
            return False, -0.3, "Too long"
        
        # Model names typically contain alphanumeric codes
        has_alphanumeric = bool(re.search(r'[A-Z0-9]{2,}', text, re.IGNORECASE))
        
        if has_alphanumeric:
            return True, +0.1, "Contains model code"
        else:
            return True, -0.1, "No clear model code"
    
    def validate_horsepower(self, value: any) -> Tuple[bool, float, str]:
        """
        Validate horsepower value.
        
        Args:
            value: Extracted HP value (int or str)
            
        Returns:
            (is_valid, confidence_adjustment, reason)
        """
        try:
            hp = int(value)
        except (ValueError, TypeError):
            return False, -0.5, "Not a number"
        
        # Typical tractor HP range
        if 10 <= hp <= 200:
            return True, +0.1, "Within typical range"
        elif 5 <= hp <= 250:
            return True, 0.0, "Acceptable range"
        else:
            return False, -0.4, f"Out of range: {hp}"
    
    def validate_asset_cost(self, value: any) -> Tuple[bool, float, str]:
        """
        Validate asset cost value.
        
        Args:
            value: Extracted cost value (int or str)
            
        Returns:
            (is_valid, confidence_adjustment, reason)
        """
        try:
            cost = int(value)
        except (ValueError, TypeError):
            return False, -0.5, "Not a number"
        
        # Typical tractor price range in INR
        if 50000 <= cost <= 5000000:
            # Check if divisible by 1000 (prices typically rounded)
            if cost % 1000 == 0:
                return True, +0.15, "Within range and rounded"
            else:
                # Non-rounded prices (like 497229) are suspicious for asset costs
                return True, -0.3, "Within range but suspiciously precise (not rounded)"
        elif 10000 <= cost <= 10000000:
            return True, -0.1, "Acceptable but unusual range"
        else:
            return False, -0.4, f"Out of range: {cost}"


# Export
__all__ = ['FieldValidator']
