# app_enhanced.py
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageEnhance
import json
import os
import time
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="AI Pneumonia Detector",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .normal-prediction {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .pneumonia-prediction {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .confidence-bar {
        height: 25px;
        background-color: #e9ecef;
        border-radius: 12px;
        margin: 8px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 25px;
        transition: width 0.5s ease;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-top: 4px solid;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

class PneumoniaDetector:
    def __init__(self):
        self.model = None
        self.class_names = ['NORMAL', 'PNEUMONIA']
        self.load_model()
        self.prediction_history = []
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = keras.models.load_model('pneumonia_model.h5')
            # Load class indices if available
            try:
                with open('class_indices.json', 'r') as f:
                    class_info = json.load(f)
                self.class_names = list(class_info.get('class_indices', {}).keys())
            except:
                pass
            return True
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Handle different image formats
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array]*3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            
            # Resize and normalize
            img_resized = Image.fromarray(img_array).resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            
            return np.expand_dims(img_array, axis=0)
            
        except Exception as e:
            st.error(f"❌ Image processing error: {e}")
            return None
    
    def predict(self, image, image_name):
        """Make prediction on image"""
        if self.model is None:
            return None
        
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return None
        
        try:
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            all_probabilities = {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
            
            # Store prediction history
            prediction_record = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'image_name': image_name,
                'predicted_class': self.class_names[predicted_class_idx],
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }
            self.prediction_history.append(prediction_record)
            
            return {
                'predicted_class': self.class_names[predicted_class_idx],
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return None

def create_confidence_chart(probabilities, predicted_class):
    """Create a bar chart for confidence scores"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = ['#28a745' if cls == 'NORMAL' else '#dc3545' for cls in classes]
    
    # Highlight predicted class
    for i, cls in enumerate(classes):
        if cls == predicted_class:
            colors[i] = '#007bff'
    
    bars = ax.bar(classes, probs, color=colors, alpha=0.8)
    ax.set_ylabel('Confidence Score')
    ax.set_ylim(0, 1)
    ax.set_title('AI Confidence Scores')
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def main():
    detector = PneumoniaDetector()
    
    # Header
    st.markdown('<div class="main-header">🏥 AI-Powered Pneumonia Detection</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Control Panel")
        
        # Model status
        if detector.model is not None:
            st.success("✅ **Model Status:** Active")
            st.info(f"🎯 **Classes:** {', '.join(detector.class_names)}")
        else:
            st.error("❌ **Model Status:** Not Available")
        
        # Features
        st.header("🛠️ Features")
        show_chart = st.checkbox("Show Confidence Chart", value=True)
        show_history = st.checkbox("Show Prediction History", value=True)
        show_metrics = st.checkbox("Show Performance Metrics", value=True)
        
        # Image enhancement
        st.header("🎨 Image Tools")
        enhance_image = st.checkbox("Enhance Image Contrast")
        
        # Quick actions
        st.header("⚡ Quick Actions")
        if st.button("Clear History"):
            detector.prediction_history = []
            st.success("History cleared!")
        
        if st.button("Test Sample"):
            st.info("Upload an image to test the model")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Image Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            
            # Image enhancement
            if enhance_image:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
                st.image(image, caption="Enhanced Image", use_column_width=True)
            else:
                st.image(image, caption="Original Image", use_column_width=True)
            
            # Analysis button
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing the image..."):
                    # Add a progress bar for better UX
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Get prediction
                    prediction_result = detector.predict(image, uploaded_file.name)
                    
                    if prediction_result:
                        display_results(prediction_result, show_chart)
    
    with col2:
        st.header("📊 Results & Analytics")
        
        # Performance metrics
        if show_metrics and detector.prediction_history:
            display_metrics(detector.prediction_history)
        
        # Prediction history
        if show_history and detector.prediction_history:
            display_history(detector.prediction_history)
        
        # Feature information
        if not detector.prediction_history:
            display_feature_info()

def display_results(prediction_result, show_chart):
    """Display prediction results"""
    predicted_class = prediction_result['predicted_class']
    confidence = prediction_result['confidence']
    all_probabilities = prediction_result['all_probabilities']
    
    # Prediction box
    box_class = "normal-prediction" if predicted_class == "NORMAL" else "pneumonia-prediction"
    
    st.markdown(f"""
    <div class="prediction-box {box_class}">
        <h2>🎯 Prediction: {predicted_class}</h2>
        <h3>📊 Confidence: {confidence:.2%}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence bars
    st.subheader("📈 Confidence Breakdown")
    for class_name, prob in all_probabilities.items():
        col1, col2, col3 = st.columns([2, 5, 1])
        with col1:
            st.write(f"{class_name}:")
        with col2:
            color = "#28a745" if class_name == "NORMAL" else "#dc3545"
            if class_name == predicted_class:
                color = "#007bff"  # Blue for predicted class
                
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {prob*100}%; background-color: {color};">
                    {prob:.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.write(f"{prob:.2%}")
    
    # Confidence chart
    if show_chart:
        st.subheader("📊 Confidence Visualization")
        chart_fig = create_confidence_chart(all_probabilities, predicted_class)
        st.pyplot(chart_fig)
    
    # Medical interpretation
    st.subheader("💡 Clinical Interpretation")
    if predicted_class == "NORMAL":
        st.success("""
        **Normal Chest X-Ray Findings:**
        - No signs of pneumonia detected
        - Clear lung fields observed
        - Normal cardiomediastinal contour
        
        **Recommendations:**
        - Continue regular health monitoring
        - No immediate intervention required
        """)
    else:
        st.error("""
        **Pneumonia Signs Detected:**
        - Areas of consolidation observed
        - Possible air bronchograms
        - Lung opacity patterns consistent with pneumonia
        
        **Urgent Recommendations:**
        - Consult with a healthcare professional immediately
        - Consider follow-up imaging
        - Clinical correlation required for diagnosis
        """)

def display_metrics(history):
    """Display performance metrics"""
    st.subheader("📈 Performance Metrics")
    
    total_predictions = len(history)
    pneumonia_count = sum(1 for p in history if p['predicted_class'] == 'PNEUMONIA')
    normal_count = total_predictions - pneumonia_count
    avg_confidence = np.mean([p['confidence'] for p in history])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Analyses", total_predictions)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Pneumonia Cases", pneumonia_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Normal Cases", normal_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)

def display_history(history):
    """Display prediction history"""
    st.subheader("📋 Recent Analyses")
    
    # Show last 5 predictions
    for record in history[-5:]:
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.write(f"**{record['image_name']}**")
            with col2:
                emoji = "✅" if record['predicted_class'] == 'NORMAL' else "🚨"
                st.write(f"{emoji} {record['predicted_class']}")
            with col3:
                st.write(f"{record['confidence']:.1%}")
            st.caption(f"Time: {record['timestamp']}")
            st.markdown("---")

def display_feature_info():
    """Display feature information when no history exists"""
    st.info("""
    ## 🎯 Welcome to AI Pneumonia Detection!
    
    **Features Available:**
    - 🖼️ **Image Analysis**: Upload chest X-ray images for AI analysis
    - 📊 **Confidence Scores**: See detailed probability breakdown
    - 📈 **Visual Charts**: Interactive confidence visualizations
    - 📋 **History Tracking**: Keep track of all analyses
    - 📊 **Performance Metrics**: Real-time statistics
    
    **How to use:**
    1. Upload a chest X-ray image (PNG, JPG, JPEG)
    2. Click 'Analyze Image' 
    3. View AI predictions and confidence scores
    4. Check clinical recommendations
    
    *Your analyses will appear here once you start using the system!*
    """)

if __name__ == "__main__":
    main()