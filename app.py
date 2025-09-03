import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Configure the page
st.set_page_config(
    page_title="FaceTime Mirror",
    page_icon="📹",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f1f1f;
        margin-bottom: 2rem;
    }
    .video-container {
        display: flex;
        justify-content: center;
        align-items: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .controls {
        text-align: center;
        margin: 20px 0;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online {
        background-color: #4CAF50;
        animation: pulse 2s infinite;
    }
    .status-offline {
        background-color: #f44336;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


# Video transformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.mirror = True
        self.filter_type = "none"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Mirror the image (flip horizontally like FaceTime)
        if self.mirror:
            img = cv2.flip(img, 1)

        # Apply filters based on selection
        if self.filter_type == "blur":
            img = cv2.GaussianBlur(img, (15, 15), 0)
        elif self.filter_type == "sepia":
            # Create sepia effect
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            img = cv2.transform(img, kernel)
        elif self.filter_type == "edge":
            # Edge detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# Main app
def main():
    st.markdown("<h1 class='main-header'>📹 FaceTime Mirror</h1>", unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")

        # Mirror toggle
        mirror_enabled = st.checkbox("Mirror Image", value=True, help="Flip image horizontally like FaceTime")

        # Filter selection
        filter_option = st.selectbox(
            "Choose Filter",
            ["none", "blur", "sepia", "edge"],
            help="Apply different visual effects"
        )

        # Status indicator
        st.markdown("### Status")
        if 'webrtc_ctx' in locals():
            st.markdown('<span class="status-indicator status-online"></span>Camera Active', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-offline"></span>Camera Inactive', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Click 'START' to begin video")
        st.markdown("2. Allow camera access when prompted")
        st.markdown("3. Use controls to adjust settings")
        st.markdown("4. Click 'STOP' to end session")

    # Create video transformer
    video_transformer = VideoTransformer()
    video_transformer.mirror = mirror_enabled
    video_transformer.filter_type = filter_option

    # Main video container
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)

        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="facetime-mirror",
            video_transformer_factory=lambda: video_transformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"min": 640, "ideal": 1280, "max": 1920},
                    "height": {"min": 480, "ideal": 720, "max": 1080},
                    "frameRate": {"min": 15, "ideal": 30, "max": 60}
                },
                "audio": False
            },
            async_processing=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # Additional info
    with st.expander("ℹ️ About This App"):
        st.write("""
        This FaceTime-style mirror app uses your webcam to create a real-time video feed.

        **Features:**
        - Real-time webcam streaming
        - Mirror mode (like FaceTime)
        - Multiple video filters
        - Responsive design
        - Privacy-focused (no data stored)

        **Technical Details:**
        - Built with Streamlit and WebRTC
        - Uses OpenCV for video processing
        - All processing happens locally in your browser
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Made with ❤️ using Streamlit • Your privacy is protected - no video data is stored"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
