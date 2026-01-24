#!/usr/bin/env python3
"""
Document AI - Invoice Field Extraction
Main executable for processing invoices and extracting structured fields.

Usage:
    python executable.py <input_path> [--output <output_path>] [--visualize]
    
Arguments:
    input_path: Path to PDF/image file or directory of files
    --output: Output JSON file path (default: sample_output/result.json)
    --visualize: Save visualization images with detected regions
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessor import load_document, preprocess_for_ocr, resize_image, deskew_image
from ocr_engine import OCREngine
from field_extractor import FieldExtractor, ExtractionResult
from visual_detector import VisualDetector
from confidence import ConfidenceScorer
from utils import list_documents, extract_doc_id, Timer, save_json


class DocumentAIPipeline:
    """
    Main pipeline for document processing and field extraction.
    """
    
    def __init__(self, config_path: str = None, invoice_type: str = 'tractor_quotation', fast_mode: bool = False):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to field configuration YAML
            invoice_type: Type of invoice to process
            fast_mode: If True, use baseline settings for faster processing
        """
        self.config_path = config_path or str(Path(__file__).parent / 'config' / 'field_config.yaml')
        self.invoice_type = invoice_type
        self.fast_mode = fast_mode
        
        # Initialize components
        print("Initializing Document AI Pipeline...")
        self.ocr_engine = OCREngine(languages=['en', 'hi', 'gu'], use_gpu=False, fast_mode=fast_mode)
        self.field_extractor = FieldExtractor(self.config_path)
        self.field_extractor.set_invoice_type(invoice_type)
        self.visual_detector = VisualDetector(use_yolo=True)
        self.confidence_scorer = ConfidenceScorer()
        
        print("Pipeline initialized successfully!")
    
    def process_document(self, file_path: str, visualize: bool = False) -> Dict[str, Any]:
        """
        Process a single document and extract all fields.
        
        Args:
            file_path: Path to document (PDF or image)
            visualize: Whether to generate visualization
            
        Returns:
            Extraction result as dictionary
        """
        doc_id = extract_doc_id(file_path)
        print(f"\nProcessing: {doc_id}")
        
        with Timer() as timer:
            try:
                # Load document
                images = load_document(file_path)
                
                if not images:
                    return {'doc_id': doc_id, 'error': 'Failed to load document'}
                
                # Process first page (most invoices are single page)
                image = images[0]
                
                # Resize if too large
                image, scale = resize_image(image, max_size=2000)
                
                # Deskew image to handle rotated documents
                image = deskew_image(image)
                
                # Run OCR
                print(f"  Running OCR...")
                ocr_data = self.ocr_engine.get_text_with_positions(image)
                
                # Extract text fields
                print(f"  Extracting fields...")
                result = self.field_extractor.extract_all_fields(ocr_data, doc_id)
                
                # Detect visual elements (signature, stamp)
                print(f"  Detecting signatures/stamps...")
                visual_detections = self.visual_detector.detect(image)
                
                # Add visual detections to result
                self.field_extractor.add_visual_detection(
                    result,
                    visual_detections.get('signature'),
                    visual_detections.get('stamp')
                )
                
                # Calculate final confidence
                field_confidences = {
                    name: field.confidence 
                    for name, field in result.fields.items()
                }
                result.overall_confidence = self.confidence_scorer.calculate_document_confidence(
                    field_confidences
                )
                
                result.processing_time_sec = time.time() - timer.start_time
                
                # Master file matching
                # Matches dealer name (fuzzy ≥90%) and model name (exact)
                try:
                    from master_matcher import MasterFileMatcher
                    matcher = MasterFileMatcher()
                    result_json = matcher.validate_and_match(result.to_json())
                except Exception as e:
                    print(f"  Warning: Master matching failed: {e}")
                    result_json = result.to_json()
                
                # Generate visualization if requested
                if visualize:
                    self._save_visualization(image, visual_detections, doc_id)
                
                # Print summary
                self._print_extraction_summary(result)
                
                return result_json
                
            except Exception as e:
                print(f"  Error: {e}")
                return {'doc_id': doc_id, 'error': str(e)}
    
    def process_directory(self, directory: str, output_path: str = None, 
                          visualize: bool = False) -> List[Dict[str, Any]]:
        """
        Process all documents in a directory.
        
        Args:
            directory: Path to directory containing documents
            output_path: Path for output JSON
            visualize: Whether to generate visualizations
            
        Returns:
            List of extraction results
        """
        files = list_documents(directory)
        print(f"Found {len(files)} documents to process")
        
        results = []
        for file_path in files:
            result = self.process_document(file_path, visualize)
            results.append(result)
        
        # Save results
        if output_path:
            save_json({'documents': results}, output_path)
            print(f"\nResults saved to: {output_path}")
        
        # Print summary
        self._print_batch_summary(results)
        
        return results
    
    def _print_extraction_summary(self, result: ExtractionResult):
        """Print extraction summary for a document."""
        print(f"  Extracted fields:")
        for name, field in result.fields.items():
            if name in ['signature', 'stamp']:
                value = f"{'Present' if field.value else 'Not found'}"
                if field.bbox:
                    value += f" (bbox: {field.bbox})"
            else:
                value = field.value
            print(f"    {name}: {value} (conf: {field.confidence:.2f})")
        print(f"  Overall confidence: {result.overall_confidence:.2f}")
    
    def _print_batch_summary(self, results: List[Dict[str, Any]]):
        """Print summary for batch processing."""
        total = len(results)
        successful = sum(1 for r in results if 'error' not in r)
        
        if successful > 0:
            avg_confidence = sum(
                r.get('confidence', 0) for r in results if 'error' not in r
            ) / successful
            avg_time = sum(
                r.get('processing_time_sec', 0) for r in results if 'error' not in r
            ) / successful
        else:
            avg_confidence = 0
            avg_time = 0
        
        print(f"\n{'='*50}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"Total documents: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"Average processing time: {avg_time:.2f}s")
        print(f"Cost per document: $0.00 (open-source)")
    
    def _save_visualization(self, image, detections: Dict, doc_id: str):
        """Save visualization image with detections."""
        import cv2
        
        viz_dir = Path(__file__).parent / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        viz_image = self.visual_detector.visualize_detections(image, detections)
        cv2.imwrite(str(viz_dir / f"{doc_id}_viz.png"), viz_image)


def main():
    parser = argparse.ArgumentParser(
        description='Document AI - Invoice Field Extraction'
    )
    parser.add_argument(
        'input_path',
        help='Path to document file or directory'
    )
    parser.add_argument(
        '--output', '-o',
        default='sample_output/result.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Save visualization images'
    )
    parser.add_argument(
        '--invoice-type', '-t',
        default='tractor_quotation',
        help='Invoice type (default: tractor_quotation)'
    )
    parser.add_argument(
        '--fast', '-f',
        action='store_true',
        help='Fast mode - use only Tesseract (2s vs 45s, slightly lower accuracy)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DocumentAIPipeline(invoice_type=args.invoice_type, fast_mode=args.fast)
    
    # Process input
    input_path = Path(args.input_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_dir():
        # Process directory
        results = pipeline.process_directory(
            str(input_path), 
            str(output_path),
            args.visualize
        )
    else:
        # Process single file
        result = pipeline.process_document(str(input_path), args.visualize)
        save_json(result, str(output_path))
        print(f"\nResult saved to: {output_path}")


if __name__ == '__main__':
    main()
