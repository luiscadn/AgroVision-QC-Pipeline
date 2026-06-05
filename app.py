from pathlib import Path
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from PIL import Image
import cv2
import av
import tempfile
from src.models.predict import predict_image

st.set_page_config(
    page_title="Agrovision QC",
    layout="centered",
)
st.title("Agrovision QC")
st.write(
    "Carga, captura, maneja en vivo una imagen de una fruta"
    "Intenta que sea un fondo simpre, el sistema predice la calidad y estima el tamaño relativo"
)

model_option = st.selectbox(
    "Modelo para la predicción",
    options = ["random_forest", "svm", "cnn"],
    format_func=lambda value:{
        "random_forest": "Random Forest",
        "svm" : "SVM",
        "cnn" : "CNN",
    }[value],
)

input_option = st.radio(
    "Entrada de la imagen",
    options = ["Cargar imagen", "Capturar con cámara", "Capturar con cámara en vivo"],
    horizontal = True,
)

image_file = None

class FruitVideo(VideoProcessorBase):
    def __init__(self):
        self.last_prediction = "Analizando..."
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
    
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                pil_image.save(tmp.name)
                result = predict_image(tmp.name, model_type=model_option)

            text = f"{result['quality'].upper()}"
            self.last_prediction = text

            cv2.putText(
                img,
                text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        except Exception:
            cv2.putText(
                img,
                "No se detecta la fruta",
                (30, 50),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


if input_option == "Cargar imagen":
    image_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png", "bmp"],
    )

elif input_option == "Capturar con cámara":
    image_file = st.camera_input("Caputura una imagen")

elif input_option == "Capturar con cámara en vivo":
    st.warning(
        "Analiza frames de la cámara del celular en tiempo real"
    )

    webrtc_streamer(
        key="fruit-live-camera",
        video_processor_factory=FruitVideo,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
    )

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Imagen de entrada", use_container_width=True)

    if st.button("Predecir calidad y tamaño", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            temp_path = Path(tmp_file.name)
            image.save(temp_path)
        
        try:
            with st.spinner("Analizando imagen..."):
                result = predict_image(temp_path, model_type=model_option)
            
            st.subheader("Resultado")
            st.success(f"Calidad estimada: **{result['quality'].upper()}**")
            st.info(f"Tamaño estimado: **{result['size'].upper()}**")
            st.write(f"Área segmentada: `{result['area_px']:.2f}` px")

            if result.get("confidence") is not None:
                st.write(f"Confianza: **{result['confidence']:.2%}**")

            if result.get("probabilities"):
                st.write("Probabilidades por clase:")
                st.json(result["probabilities"])
        except Exception as error:
            st.error("No se pudo generar la predicción.")
            st.exception(error)
