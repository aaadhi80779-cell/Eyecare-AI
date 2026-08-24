import streamlit as st
import os
import uuid
import hashlib
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EyeCare AI",
    page_icon="👁️",
    layout="centered"
)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "eye_model.keras"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

RESULT_FILE = os.path.join(
    DATA_DIR,
    "patients.csv"
)

os.makedirs(DATA_DIR, exist_ok=True)


CLASS_NAMES = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferative_DR",
    "Severe"
]


# =========================================================
# SESSION STATE
# =========================================================

if "patient_id" not in st.session_state:
    st.session_state.patient_id = (
        "P-" + uuid.uuid4().hex[:8].upper()
    )

if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None

if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None


# =========================================================
# LOAD AI MODEL
# =========================================================

model = None
model_loaded = False
model_error = ""

if os.path.exists(MODEL_PATH):

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        model_loaded = True

    except Exception as e:

        model = None
        model_loaded = False
        model_error = str(e)

else:

    model_error = (
        "eye_model.keras not found inside models folder."
    )


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    img = cv2.resize(
        image,
        (224, 224)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(
        np.float32
    ) / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    return img


# =========================================================
# AI PREDICTION
# =========================================================

def predict_image(image):

    if model is None:

        return None, 0.0, None

    processed = preprocess_image(
        image
    )

    prediction = model.predict(
        processed,
        verbose=0
    )

    raw_output = np.asarray(
        prediction
    ).flatten()

    # -----------------------------------------------------
    # MODEL DEBUG INFORMATION
    # -----------------------------------------------------

    st.write(
        "### Model Output"
    )

    st.write(
        "Model Input Shape:",
        processed.shape
    )

    st.write(
        "Output Shape:",
        raw_output.shape
    )

    st.write(
        "Raw Model Output:",
        raw_output
    )

    if raw_output.size == 0:

        return (
            None,
            0.0,
            raw_output
        )

    # =====================================================
    # BINARY MODEL
    # =====================================================

    if raw_output.size == 1:

        value = float(
            raw_output[0]
        )

        if (
            value < 0.0
            or
            value > 1.0
        ):

            probability = (
                1.0 /
                (
                    1.0 +
                    np.exp(-value)
                )
            )

        else:

            probability = value

        probability = float(
            np.clip(
                probability,
                0.0,
                1.0
            )
        )

        if probability >= 0.5:

            result = (
                "Diabetic Retinopathy Suspected"
            )

            confidence = (
                probability * 100
            )

        else:

            result = (
                "No Diabetic Retinopathy Detected"
            )

            confidence = (
                (1.0 - probability) * 100
            )

        return (
            result,
            confidence,
            np.array([probability])
        )

    # =====================================================
    # MULTI CLASS MODEL
    # =====================================================

    values = raw_output.astype(
        np.float64
    )

    # -----------------------------------------------------
    # CHECK WHETHER OUTPUT IS PROBABILITY
    # -----------------------------------------------------

    probability_like = (
        np.all(values >= 0)
        and
        np.all(values <= 1)
        and
        np.isclose(
            np.sum(values),
            1.0,
            atol=0.05
        )
    )

    # -----------------------------------------------------
    # CONVERT LOGITS TO PROBABILITIES
    # -----------------------------------------------------

    if not probability_like:

        exp_values = np.exp(
            values -
            np.max(values)
        )

        probabilities = (
            exp_values /
            np.sum(exp_values)
        )

    else:

        probabilities = values

    # -----------------------------------------------------
    # DISPLAY EACH CLASS PROBABILITY
    # -----------------------------------------------------

    st.write(
        "### Class Probabilities"
    )

    probability_data = []

    for i in range(
        min(
            len(probabilities),
            len(CLASS_NAMES)
        )
    ):

        probability_data.append(
            {
                "Class":
                    CLASS_NAMES[i],

                "Probability":
                    f"{float(probabilities[i]) * 100:.2f}%"
            }
        )

    probability_df = pd.DataFrame(
        probability_data
    )

    st.table(
        probability_df
    )

    # -----------------------------------------------------
    # FIND HIGHEST PROBABILITY
    # -----------------------------------------------------

    index = int(
        np.argmax(
            probabilities
        )
    )

    if index >= len(CLASS_NAMES):

        index = (
            len(CLASS_NAMES) - 1
        )

    result = CLASS_NAMES[
        index
    ]

    confidence = float(
        probabilities[index] * 100
    )

    return (
        result,
        confidence,
        probabilities
    )


# =========================================================
# RECOMMENDATION
# =========================================================

def get_recommendation(result):

    recommendations = {

        "No_DR":
            "No diabetic retinopathy category was identified by the screening model. Routine eye-care follow-up is recommended.",

        "Mild":
            "Mild diabetic retinopathy category identified. Professional eye-care follow-up is recommended.",

        "Moderate":
            "Moderate diabetic retinopathy category identified. Professional eye examination and follow-up are recommended.",

        "Severe":
            "Severe diabetic retinopathy category identified. Prompt professional eye-care evaluation is recommended.",

        "Proliferative_DR":
            "Proliferative diabetic retinopathy category identified. Prompt specialist eye-care evaluation is recommended.",

        "Diabetic Retinopathy Suspected":
            "The screening model indicates possible diabetic retinopathy. Professional eye-care evaluation is recommended.",

        "No Diabetic Retinopathy Detected":
            "The screening model did not identify diabetic retinopathy. Routine eye-care follow-up is recommended."
    }

    return recommendations.get(
        result,
        "Professional eye-care review is recommended."
    )


# =========================================================
# SAVE RESULT
# =========================================================

def save_result(report):

    new_data = pd.DataFrame(
        [report]
    )

    if os.path.exists(
        RESULT_FILE
    ):

        try:

            old_data = pd.read_csv(
                RESULT_FILE
            )

            new_data = pd.concat(
                [
                    old_data,
                    new_data
                ],
                ignore_index=True
            )

        except Exception:

            pass

    new_data.to_csv(
        RESULT_FILE,
        index=False
    )


# =========================================================
# PDF REPORT
# =========================================================

def create_patient_pdf(report):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=12,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8
    )

    story = []

    story.append(
        Paragraph(
            "EYECARE AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Diabetic Retinopathy Screening Report",
            subtitle_style
        )
    )

    # =====================================================
    # PATIENT DETAILS
    # =====================================================

    story.append(
        Paragraph(
            "Patient Details",
            heading_style
        )
    )

    patient_data = [

        [
            "Patient ID",
            str(report["Patient ID"])
        ],

        [
            "Patient Name",
            str(report["Patient Name"])
        ],

        [
            "Age",
            str(report["Age"])
        ],

        [
            "Gender",
            str(report["Gender"])
        ]
    ]

    patient_table = Table(
        patient_data,
        colWidths=[150, 330]
    )

    patient_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ]
        )
    )

    story.append(
        patient_table
    )

    # =====================================================
    # DIABETES INFORMATION
    # =====================================================

    story.append(
        Paragraph(
            "Diabetes Information",
            heading_style
        )
    )

    diabetes_data = [

        [
            "Diabetes Status",
            str(
                report[
                    "Diabetes Status"
                ]
            )
        ],

        [
            "Diabetes Type",
            str(
                report[
                    "Diabetes Type"
                ]
            )
        ],

        [
            "Duration",
            str(
                report[
                    "Diabetes Duration"
                ]
            )
        ],

        [
            "HbA1c",
            str(
                report[
                    "HbA1c"
                ]
            )
        ],

        [
            "Treatment",
            str(
                report[
                    "Diabetes Treatment"
                ]
            )
        ],

        [
            "Previous Eye Problem",
            str(
                report[
                    "Previous Eye Problem"
                ]
            )
        ]
    ]

    diabetes_table = Table(
        diabetes_data,
        colWidths=[150, 330]
    )

    diabetes_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ]
        )
    )

    story.append(
        diabetes_table
    )

    # =====================================================
    # SCREENING RESULT
    # =====================================================

    story.append(
        Paragraph(
            "Screening Result",
            heading_style
        )
    )

    screening_data = [

        [
            "AI Result",
            str(
                report[
                    "Predicted Class"
                ]
            )
        ],

        [
            "Confidence",
            f'{float(report["Confidence"]):.2f}%'
        ],

        [
            "Image Quality",
            str(
                report[
                    "Image Quality"
                ]
            )
        ],

        [
            "Referral",
            str(
                report[
                    "Referral Status"
                ]
            )
        ],

        [
            "Date & Time",
            str(
                report[
                    "Date & Time"
                ]
            )
        ]
    ]

    screening_table = Table(
        screening_data,
        colWidths=[150, 330]
    )

    screening_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ]
        )
    )

    story.append(
        screening_table
    )

    # =====================================================
    # RECOMMENDATION
    # =====================================================

    story.append(
        Paragraph(
            "Recommendation",
            heading_style
        )
    )

    story.append(
        Paragraph(
            str(
                report[
                    "Recommendation"
                ]
            ),
            styles["Normal"]
        )
    )

    story.append(
        Spacer(
            1,
            20
        )
    )

    # =====================================================
    # DISCLAIMER
    # =====================================================

    story.append(
        Paragraph(
            "Disclaimer",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "EyeCare AI is a research and educational screening prototype. The AI result is not a medical diagnosis and should not replace examination by a qualified eye-care professional.",
            styles["Normal"]
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# TITLE
# =========================================================

st.title(
    "👁️ EyeCare AI"
)

st.write(
    "AI-Assisted Diabetic Retinopathy Screening System"
)


# =========================================================
# MODEL STATUS
# =========================================================

if model_loaded:

    st.success(
        "✅ AI model loaded successfully."
    )

else:

    st.warning(
        "⚠️ AI model is not available."
    )

    st.caption(
        "Place a compatible eye_model.keras file inside the models folder."
    )

    if model_error:

        st.code(
            model_error
        )


# =========================================================
# PATIENT REGISTRATION
# =========================================================

st.divider()

st.header(
    "1. Register Patient"
)

patient_id = (
    st.session_state.patient_id
)

st.text_input(
    "Patient ID",
    value=patient_id,
    disabled=True
)

patient_name = st.text_input(
    "Patient Name"
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=18
)

gender = st.selectbox(
    "Gender",
    [
        "Select",
        "Female",
        "Male",
        "Other"
    ]
)

if st.button(
    "Register Patient"
):

    if not patient_name.strip():

        st.error(
            "Please enter patient name."
        )

    elif gender == "Select":

        st.error(
            "Please select gender."
        )

    else:

        st.success(
            f"Patient {patient_id} registered successfully!"
        )


# =========================================================
# DIABETES INFORMATION
# =========================================================

st.divider()

st.header(
    "2. Basic Diabetes Information"
)

diabetes_status = st.selectbox(
    "Diabetes Status",
    [
        "Select",
        "Yes",
        "No",
        "Unknown"
    ]
)

if diabetes_status == "Yes":

    diabetes_type = st.selectbox(
        "Type of Diabetes",
        [
            "Select",
            "Type 1",
            "Type 2",
            "Gestational",
            "Other"
        ]
    )

    diabetes_duration = st.number_input(
        "Duration of Diabetes (years)",
        min_value=0,
        max_value=100,
        value=0
    )

    hba1c = st.number_input(
        "Latest HbA1c (%)",
        min_value=0.0,
        max_value=30.0,
        value=0.0,
        step=0.1
    )

    treatment = st.multiselect(
        "Current Diabetes Treatment",
        [
            "Diet control",
            "Oral medication",
            "Insulin",
            "Other"
        ]
    )

    previous_eye_problem = st.selectbox(
        "Previous Diabetic Eye Problem",
        [
            "Select",
            "Yes",
            "No",
            "Unknown"
        ]
    )

else:

    diabetes_type = (
        "Not applicable"
    )

    diabetes_duration = 0

    hba1c = 0.0

    treatment = []

    previous_eye_problem = (
        "Not applicable"
    )


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.divider()

st.header(
    "3. Capture / Upload Retinal Image"
)

st.info(
    "📷 For demonstration, upload a retinal/fundus image. Reliable clinical screening requires an appropriate retinal imaging device."
)

source = st.radio(
    "Choose image source",
    [
        "📁 Upload Image",
        "📷 Capture with Camera"
    ],
    horizontal=True
)

if source == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload retinal image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="retinal_upload"
    )

else:

    uploaded_file = st.camera_input(
        "📷 Capture retinal image",
        key="retinal_camera"
    )


# =========================================================
# IMAGE PROCESSING
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # READ IMAGE FILE
    # -----------------------------------------------------

    file_bytes_raw = (
        uploaded_file.getvalue()
    )

    # -----------------------------------------------------
    # IMAGE HASH
    # -----------------------------------------------------

    image_hash = hashlib.md5(
        file_bytes_raw
    ).hexdigest()

    # -----------------------------------------------------
    # NEW IMAGE CHECK
    # -----------------------------------------------------

    if (
        st.session_state.last_image_hash
        != image_hash
    ):

        st.session_state.predicted_class = None

        st.session_state.confidence = None

        st.session_state.probabilities = None

        st.session_state.last_image_hash = (
            image_hash
        )

    # -----------------------------------------------------
    # DECODE IMAGE
    # -----------------------------------------------------

    file_bytes = np.frombuffer(
        file_bytes_raw,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "❌ Image read panna mudiyala. Please upload JPG/PNG image."
        )

    else:

        # =================================================
        # DISPLAY IMAGE
        # =================================================

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            rgb_image,
            caption="Uploaded Retinal Image",
            use_container_width=True
        )

        # =================================================
        # IMAGE QUALITY
        # =================================================

        st.divider()

        st.header(
            "4. Image Quality Check"
        )

        height, width = image.shape[:2]

        st.write(
            f"**Image Size:** {width} × {height} pixels"
        )

        resolution_ok = (
            width >= 224
            and
            height >= 224
        )

        gray_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        blur_score = float(
            cv2.Laplacian(
                gray_image,
                cv2.CV_64F
            ).var()
        )

        blur_ok = (
            blur_score >= 50
        )

        quality_ok = (
            resolution_ok
            and
            blur_ok
        )

        st.write(
            f"**Sharpness Score:** {blur_score:.2f}"
        )

        if resolution_ok:

            st.success(
                "✅ Resolution check passed."
            )

        else:

            st.warning(
                "⚠️ Image resolution is too low."
            )

        if blur_ok:

            st.success(
                "✅ Sharpness check passed."
            )

        else:

            st.warning(
                "⚠️ Image may be blurry."
            )

        if quality_ok:

            image_quality = "Good"

            st.success(
                "🛡️ Image quality gate passed."
            )

        else:

            image_quality = "Poor"

            st.warning(
                "⚠️ Please upload a clearer retinal image."
            )

        # =================================================
        # AI ANALYSIS
        # =================================================

        st.divider()

        st.header(
            "5. AI Analysis"
        )

        if not model_loaded:

            st.error(
                "❌ AI model is not available."
            )

            st.info(
                "Put eye_model.keras inside the models folder."
            )

        elif not quality_ok:

            st.info(
                "Please upload a clearer retinal image before AI analysis."
            )

        else:

            if st.button(
                "🔍 Analyze Retinal Image",
                key="analyze_button"
            ):

                with st.spinner(
                    "AI model is analyzing the retinal image..."
                ):

                    try:

                        (
                            predicted_class,
                            confidence,
                            raw_prediction
                        ) = predict_image(
                            image
                        )

                        if predicted_class is None:

                            st.error(
                                "Model returned no prediction."
                            )

                        else:

                            st.session_state.predicted_class = (
                                predicted_class
                            )

                            st.session_state.confidence = (
                                confidence
                            )

                            st.session_state.probabilities = (
                                raw_prediction
                            )

                            st.success(
                                "✅ AI analysis completed."
                            )

                    except Exception as e:

                        st.error(
                            "❌ AI prediction failed."
                        )

                        st.code(
                            str(e)
                        )

        # =================================================
        # SCREENING RESULT
        # =================================================

        if (
            st.session_state.predicted_class
            is not None
        ):

            st.divider()

            st.header(
                "6. Diabetic Retinopathy Screening Result"
            )

            predicted_class = (
                st.session_state.predicted_class
            )

            confidence = (
                st.session_state.confidence
            )

            # -------------------------------------------------
            # DISPLAY RESULT
            # -------------------------------------------------

            if predicted_class == "No_DR":

                display_result = (
                    "No Diabetic Retinopathy Detected"
                )

                referral = (
                    "Routine eye-care follow-up"
                )

            elif predicted_class in [
                "Mild",
                "Moderate",
                "Severe",
                "Proliferative_DR"
            ]:

                display_result = (
                    "Diabetic Retinopathy - "
                    +
                    predicted_class.replace(
                        "_",
                        " "
                    )
                )

                referral = (
                    "Professional eye-care review recommended"
                )

            else:

                display_result = (
                    predicted_class
                )

                referral = (
                    "Professional eye-care review recommended"
                )

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.subheader(
                "Result"
            )

            st.success(
                f"🩺 {display_result}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.info(
                "This is an AI-based screening result for prototype/research purposes only. It is not a medical diagnosis."
            )

            st.write(
                f"**Referral:** {referral}"
            )

            # -------------------------------------------------
            # RECOMMENDATION
            # -------------------------------------------------

            recommendation = get_recommendation(
                predicted_class
            )

            st.subheader(
                "Recommendation"
            )

            st.write(
                recommendation
            )

            # =================================================
            # REPORT DATA
            # =================================================

            report = {

                "Patient ID":
                    patient_id,

                "Patient Name":
                    patient_name,

                "Age":
                    age,

                "Gender":
                    gender,

                "Diabetes Status":
                    diabetes_status,

                "Diabetes Type":
                    diabetes_type,

                "Diabetes Duration":
                    diabetes_duration,

                "HbA1c":
                    hba1c,

                "Diabetes Treatment":
                    (
                        ", ".join(
                            treatment
                        )
                        if treatment
                        else "None"
                    ),

                "Previous Eye Problem":
                    previous_eye_problem,

                "Predicted Class":
                    display_result,

                "Confidence":
                    confidence,

                "Image Quality":
                    image_quality,

                "Referral Status":
                    referral,

                "Date & Time":
                    datetime.now().strftime(
                        "%d-%m-%Y %H:%M:%S"
                    ),

                "Recommendation":
                    recommendation
            }

            # =================================================
            # REPORT
            # =================================================

            st.divider()

            st.header(
                "7. Report"
            )

            col1, col2 = st.columns(
                2
            )

            with col1:

                if st.button(
                    "💾 Save Screening Result",
                    key="save_result_button"
                ):

                    try:

                        save_result(
                            report
                        )

                        st.success(
                            "✅ Screening result saved."
                        )

                    except Exception as e:

                        st.error(
                            "Could not save result."
                        )

                        st.code(
                            str(e)
                        )

            with col2:

                try:

                    pdf_data = create_patient_pdf(
                        report
                    )

                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_data,
                        file_name=(
                            "EyeCare_Report_"
                            +
                            patient_id
                            +
                            ".pdf"
                        ),
                        mime="application/pdf",
                        key="download_pdf_button"
                    )

                except Exception as e:

                    st.error(
                        "PDF creation failed."
                    )

                    st.code(
                        str(e)
                    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "EyeCare AI | Research & Educational Prototype | "
    "AI screening is not a substitute for professional medical examination."
)