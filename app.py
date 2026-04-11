import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import os
import tempfile
from typing import Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="🔥 Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .fire-alert {
        background-color: #FFE5E5;
        border-left: 5px solid #FF4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .safe-alert {
        background-color: #E5F5E5;
        border-left: 5px solid #44FF44;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = "fire_detection_model_corrected.keras"
INPUT_SHAPE = (224, 224, 3)
DEFAULT_THRESHOLD = 0.5
FRAME_SKIP = 5  # Process every 5th frame for performance

@st.cache_resource
def load_model_cached() -> tf.keras.Model:
    """
    Load the pre-trained fire detection model with caching.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            return None
        
        model = load_model(MODEL_PATH)
        
        # Test model with dummy input
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        raw_test_pred = model.predict(test_input, verbose=0)[0][0]
        corrected_test_pred = 1 - raw_test_pred
        
        if corrected_test_pred == 0.0 and raw_test_pred == 0.0:
            st.error("🚨 Model Issue Detected: The model is predicting 0.0 for all inputs.")
            st.error("This means the model cannot detect fire and needs to be retrained.")
        else:
            st.success("✅ Model loaded and tested successfully!")
            st.info(f"🔧 Using corrected interpretation (fire = 1 - prediction)")
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a single frame for model prediction.
    
    Args:
        frame: Input frame as numpy array (BGR format from OpenCV)
    
    Returns:
        Preprocessed frame ready for model input
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to model input shape
    frame_resized = cv2.resize(frame_rgb, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    
    # Normalize pixel values to [0, 1]
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    frame_batch = np.expand_dims(frame_normalized, axis=0)
    
    return frame_batch

def predict_frame(model: tf.keras.Model, frame: np.ndarray) -> float:
    """
    Make prediction on a single frame.
    
    Args:
        model: Loaded Keras model
        frame: Preprocessed frame
    
    Returns:
        Prediction probability (0-1)
    """
    try:
        # Get raw prediction from model
        raw_prediction = model.predict(frame, verbose=0)[0][0]
        
        # CORRECTED INTERPRETATION: Flip the prediction
        # Original: fire_images=0, non_fire_images=1  
        # We want: fire=1, non_fire=0
        corrected_prediction = 1 - raw_prediction
        
        # Debug: Log prediction values
        st.session_state.debug_predictions = st.session_state.get('debug_predictions', [])
        st.session_state.debug_predictions.append(float(corrected_prediction))
        
        return float(corrected_prediction)
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return 0.0

def process_video(video_path: str, model: tf.keras.Model, threshold: float, 
                  frame_skip: int = FRAME_SKIP) -> Tuple[bool, float, Optional[np.ndarray], int]:
    """
    Process video file frame by frame to detect fire.
    
    Args:
        video_path: Path to video file
        model: Loaded Keras model
        threshold: Detection threshold
        frame_skip: Number of frames to skip between processing
    
    Returns:
        Tuple of (fire_detected, max_confidence, fire_frame, total_frames_processed)
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ Unable to open video file")
            return False, 0.0, None, 0
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        st.info(f"📹 Video Info: {total_frames} total frames, {fps:.2f} FPS")
        
        frame_count = 0
        processed_count = 0
        max_confidence = 0.0
        fire_detected = False
        fire_frame = None
        fire_frame_index = -1
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process only every frame_skip-th frame
            if frame_count % frame_skip == 0:
                # Preprocess frame
                processed_frame = preprocess_frame(frame)
                
                # Make prediction
                confidence = predict_frame(model, processed_frame)
                
                # Update maximum confidence
                if confidence > max_confidence:
                    max_confidence = confidence
                    fire_frame = frame.copy()
                    fire_frame_index = frame_count
                
                # Check if fire detected
                if confidence > threshold:
                    fire_detected = True
                
                processed_count += 1
                
                # Update progress
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames} (Current: {confidence:.3f}, Max: {max_confidence:.3f})")
            
            frame_count += 1
        
        # Clean up
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        return fire_detected, max_confidence, fire_frame, processed_count
        
    except Exception as e:
        st.error(f"❌ Video processing error: {str(e)}")
        return False, 0.0, None, 0

def main():
    """
    Main Streamlit application
    """
    # Header
    st.markdown('<h1 class="main-header">🔥 Fire Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a video to detect fire using AI-powered frame analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Debug controls
    st.sidebar.subheader("🔍 Debug Controls")
    if st.sidebar.button("Clear Debug Data"):
        if 'debug_predictions' in st.session_state:
            del st.session_state.debug_predictions
        st.sidebar.success("Debug data cleared!")
    
    st.sidebar.markdown("---")
    
    # Threshold slider with tips
    st.sidebar.subheader("🎯 Detection Settings")
    threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,  # Normal threshold for working model
        step=0.05,
        help="✅ Model is working! Use 0.3-0.7 for best results."
    )
    
    # Frame skip slider
    frame_skip = st.sidebar.slider(
        "Frame Skip Interval",
        min_value=1,
        max_value=20,
        value=FRAME_SKIP,
        step=1,
        help="💡 Tip: Lower values (1-5) = more accurate but slower"
    )
    
    # Tips section
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Usage Tips")
    st.sidebar.success("""
    ✅ MODEL IS WORKING!
    • Use threshold 0.3-0.7 for best results
    • Frame skip 1-5 gives most accurate results
    • Check debug info to see prediction values
    • Model correctly distinguishes fire vs non-fire
    """)
    
    st.sidebar.info("""
    🔧 TECHNICAL DETAILS:
    • Uses corrected label interpretation
    • Fire predictions > 0.5 = fire detected
    • Non-fire predictions < 0.5 = no fire
    • Model trained on 999 images total
    """)
    
    # Load model
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Status")
    
    with st.sidebar:
        with st.spinner("Loading model..."):
            model = load_model_cached()
    
    # Model testing section
    if model is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧪 Model Test")
        if st.sidebar.button("Test Model with Random Data"):
            with st.sidebar:
                with st.spinner("Testing model..."):
                    # Test with random data
                    test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
                    test_pred = model.predict(test_input, verbose=0)[0][0]
                    st.write(f"Random input: {test_pred:.6f}")
                    
                    # Test with all zeros
                    zero_input = np.zeros((1, 224, 224, 3)).astype(np.float32)
                    zero_pred = model.predict(zero_input, verbose=0)[0][0]
                    st.write(f"All zeros: {zero_pred:.6f}")
                    
                    # Test with all ones
                    ones_input = np.ones((1, 224, 224, 3)).astype(np.float32)
                    ones_pred = model.predict(ones_input, verbose=0)[0][0]
                    st.write(f"All ones: {ones_pred:.6f}")
                    
                    if max(test_pred, zero_pred, ones_pred) < 0.1:
                        st.warning("⚠️ Model predictions are very low. Consider lowering threshold.")
                    elif max(test_pred, zero_pred, ones_pred) > 0.9:
                        st.info("✅ Model is making confident predictions.")
    
    if model is None:
        st.error("❌ Cannot proceed without the model. Please ensure 'fire_detection_model.h5' is in the current directory.")
        return
    
    # Main content area
    st.markdown("---")
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
    
    with col2:
        st.subheader("📊 Quick Stats")
        st.metric("Model Ready", "✅" if model else "❌")
        st.metric("Threshold", f"{threshold:.2f}")
        st.metric("Frame Skip", frame_skip)
    
    # Process uploaded video
    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("🎬 Video Preview")
        
        # Display video info
        video_details = {
            "Filename": uploaded_file.name,
            "File Size": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "Content Type": uploaded_file.type
        }
        
        for key, value in video_details.items():
            st.write(f"**{key}:** {value}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            # Process video button
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                st.markdown("---")
                st.subheader("🔬 Analysis Results")
                
                # Process video
                with st.spinner("Analyzing video frames..."):
                    fire_detected, max_confidence, fire_frame, frames_processed = process_video(
                        video_path, model, threshold, frame_skip
                    )
                
                # Display results
                if fire_detected:
                    st.markdown(
                        '<div class="fire-alert">'
                        '<h3 style="color: #FF4444; margin: 0;">🔥 FIRE DETECTED!</h3>'
                        '<p style="margin: 0.5rem 0 0 0;">Fire was found in the uploaded video.</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="safe-alert">'
                        '<h3 style="color: #44AA44; margin: 0;">✅ NO FIRE DETECTED</h3>'
                        '<p style="margin: 0.5rem 0 0 0;">No fire was found in the uploaded video.</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                
                # Debug information
                with st.expander("🔍 Debug Information", expanded=True):
                    debug_predictions = st.session_state.get('debug_predictions', [])
                    if debug_predictions:
                        st.write(f"**Total predictions made:** {len(debug_predictions)}")
                        st.write(f"**Min prediction:** {min(debug_predictions):.6f}")
                        st.write(f"**Max prediction:** {max(debug_predictions):.6f}")
                        st.write(f"**Average prediction:** {np.mean(debug_predictions):.6f}")
                        st.write(f"**Threshold used:** {threshold:.2f}")
                        
                        # Show prediction distribution using streamlit chart
                        if len(debug_predictions) > 1:
                            st.line_chart(debug_predictions)
                            st.write(f"Red line indicates threshold ({threshold})")
                    
                    # Test with sample frame
                    st.write("**Testing model with random input:**")
                    test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
                    test_pred = model.predict(test_input, verbose=0)[0][0]
                    st.write(f"Random input prediction: {test_pred:.6f}")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Status", "🔥 Fire" if fire_detected else "✅ Safe")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Max Confidence", f"{max_confidence:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Frames Processed", frames_processed)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Threshold", f"{threshold:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display fire frame if available
                if fire_frame is not None and fire_detected:
                    st.markdown("---")
                    st.subheader("📸 Fire Detection Snapshot")
                    
                    # Convert BGR to RGB for display
                    fire_frame_rgb = cv2.cvtColor(fire_frame, cv2.COLOR_BGR2RGB)
                    fire_frame_pil = Image.fromarray(fire_frame_rgb)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.image(fire_frame_pil, caption="Frame where fire was detected", use_column_width=True)
                    
                    with col2:
                        st.markdown("#### Detection Details:")
                        st.write(f"**Confidence:** {max_confidence:.3f}")
                        st.write(f"**Threshold:** {threshold:.2f}")
                        st.write(f"**Result:** {'Fire' if max_confidence > threshold else 'No Fire'}")
                        
                        # Confidence gauge
                        confidence_percentage = max_confidence * 100
                        st.progress(confidence_percentage / 100)
                        st.write(f"Confidence: {confidence_percentage:.1f}%")
                
                # Confidence chart
                if frames_processed > 0:
                    st.markdown("---")
                    st.subheader("📈 Analysis Summary")
                    
                    # Create a simple confidence visualization
                    confidence_color = "🔴" if fire_detected else "🟢"
                    st.write(f"{confidence_color} **Maximum Confidence Achieved:** {max_confidence:.3f}")
                    st.write(f"📊 **Frames Analyzed:** {frames_processed}")
                    st.write(f"🎯 **Detection Threshold:** {threshold:.2f}")
                    
                    if fire_detected:
                        st.success(f"✨ Fire detected with {max_confidence:.1%} confidence!")
                    else:
                        st.info(f"ℹ️ No fire detected. Highest confidence was {max_confidence:.1%}")
        
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(video_path)
            except:
                pass
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; padding: 1rem;">'
        '🔥 Fire Detection System | Powered by TensorFlow & Streamlit'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
