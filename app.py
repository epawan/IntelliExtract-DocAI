import streamlit as st
import os
import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import sys
import time
import base64

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from executable import DocumentAIPipeline
from utils import save_json

# --- Premium UI Configurations ---
st.set_page_config(
    page_title="Intelligent Document AI Dashboard",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2227;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e444b;
    }
    .stAlert {
        border-radius: 10px;
    }
    .highlight {
        color: #00d4ff;
        font-weight: bold;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # --- Sidebar ---
    st.sidebar.image("https://img.icons8.com/color/96/tractor.png", width=80)
    st.sidebar.title("Pipeline Settings")
    st.sidebar.markdown("---")
    
    fast_mode = st.sidebar.toggle("Fast Mode (CPU-Optimized)", value=True, help="Uses Tesseract primarily. Disable for full PaddleOCR accuracy.")
    invoice_type = st.sidebar.selectbox("Document Type", ["Tractor Quotation", "Retail Invoice", "Generic Invoice"])
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("System Info")
    st.sidebar.info("Backend: PaddleOCR + YOLOv8\nLanguage: EN, HI, GU\nDevice: CPU")

    # --- Header ---
    st.title("🚜 Intelligent Document AI Dashboard")
    st.markdown("Automated field extraction for **Intelligent Banking Automation**.")
    
    # --- Main Interface ---
    uploaded_file = st.file_uploader("Drop your document here (PNG, JPG, PDF)", type=['png', 'jpg', 'jpeg'])

    # Initialize Pipeline (Cached)
    @st.cache_resource
    def get_pipeline(fast):
        return DocumentAIPipeline(fast_mode=fast)

    pipeline = get_pipeline(fast_mode)

    if uploaded_file:
        col1, col2 = st.columns([1, 1], gap="large")
        
        # Original Image Load
        image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with col1:
            st.subheader("📄 Input Document")
            st.image(image_rgb, use_container_width=True)

        # Extraction Logic
        if st.button("🚀 Run Extraction Pipeline", use_container_width=True):
            with st.spinner("Decoding Layout & Extracting Fields..."):
                start_time = time.time()
                
                # Save temp
                temp_path = f"temp_{uploaded_file.name}"
                cv2.imwrite(temp_path, image)
                
                try:
                    result = pipeline.process_document(temp_path)
                    fields = result.get('fields', {})
                    
                    with col2:
                        st.subheader("💡 Extraction Intelligence")
                        
                        # Metrics Row
                        m1, m2, m3 = st.columns(3)
                        conf = result.get('confidence', 0)
                        m1.metric("Confidence", f"{conf:.1%}")
                        m2.metric("Latency", f"{result.get('processing_time_sec', 0):.2f}s")
                        m3.metric("Cost", "$0.00", delta_color="normal")

                        st.markdown("---")

                        # Entity Table
                        st.write("### Extracted Entities")
                        
                        entities = [
                            ("🏛️ Dealer Name", fields.get('dealer_name', 'Not Specified')),
                            ("🚜 Model Name", fields.get('model_name', 'Not Specified')),
                            ("⚡ Horse Power", f"{fields.get('horse_power', 'N/A')} HP"),
                            ("💰 Asset Cost", f"₹ {fields.get('asset_cost', '0'):,}")
                        ]
                        
                        for label, value in entities:
                            st.markdown(f"**{label}**: {value}")

                        st.markdown("---")
                        
                        # Visual Verification
                        st.write("### Visual Verification")
                        v_col1, v_col2 = st.columns(2)
                        
                        sig = fields.get('signature', {})
                        stamp = fields.get('stamp', {})
                        
                        with v_col1:
                            if sig.get('present'):
                                st.success("✅ Signature Found")
                            else:
                                st.error("❌ Signature Missing")
                        
                        with v_col2:
                            if stamp.get('present'):
                                st.success("✅ Stamp Found")
                            else:
                                st.error("❌ Stamp Missing")

                        # Detection Overlay
                        with st.expander("View Detection Overlays", expanded=True):
                            viz_image = image_rgb.copy()
                            # Draw Signature
                            if sig.get('present') and sig.get('bbox'):
                                x, y, w, h = sig['bbox']
                                cv2.rectangle(viz_image, (x, y), (x+w, y+h), (0, 255, 0), 4)
                                cv2.putText(viz_image, "SIGNATURE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Draw Stamp
                            if stamp.get('present') and stamp.get('bbox'):
                                x, y, w, h = stamp['bbox']
                                cv2.rectangle(viz_image, (x, y), (x+w, y+h), (0, 150, 255), 4)
                                cv2.putText(viz_image, "STAMP", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)
                            
                            st.image(viz_image, use_container_width=True)

                        # Raw Data Toggle
                        if st.checkbox("Show Technical JSON Response"):
                            st.json(result)

                except Exception as e:
                    st.error(f"Pipeline Error: {str(e)}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    else:
        # Welcome Section
        st.markdown("---")
        st.write("### How it Works")
        step1, step2, step3 = st.columns(3)
        with step1:
            st.write("🔍 **1. Preprocessing**")
            st.caption("Auto-deskewing, noise reduction and grayscale conversion for optimal OCR.")
        with step2:
            st.write("🤖 **2. Computer Vision**")
            st.caption("YOLOv8 detects signatures/stamps while PaddleOCR extracts multilingual text.")
        with step3:
            st.write("🏗️ **3. Spatial Reasoning**")
            st.caption("Our proprietary layout engine maps labels to values across 500+ document variants.")

if __name__ == "__main__":
    main()
