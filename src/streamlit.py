import streamlit as st
from Preprocessing import preprocess_image, load_model, detect_disease

model = load_model()


def show_medical_info(diagnosis):
    st.markdown("---")
    st.markdown("### 🩺 Medical Information")
    
    if diagnosis == "Tumor":
        st.error("**Tumor Detected** 🧠")
        st.markdown("""
        - **Description**: A tumor is an abnormal growth of tissue. It can be benign (non-cancerous) or malignant (cancerous).
        - **Symptoms**: May include pain, swelling, or abnormal function of the organ.
        - **Next Steps**: Further medical imaging and biopsy may be needed to determine type and treatment.
        - **Advice**:
            - Consult a specialized oncologist.
            - Follow a healthy diet and avoid processed food.
            - Do not delay further tests.
        """)
    
    elif diagnosis == "Stone":
        st.error("**Kidney Stone Detected** 🪨")
        st.markdown("""
        - **Description**: Kidney stones are hard deposits made of minerals and salts that form inside your kidneys.
        - **Symptoms**: Severe side/back pain, blood in urine, nausea.
        - **Next Steps**: Medical evaluation, possible ultrasound or CT scan.
        - **Advice**:
            - Drink plenty of water daily.
            - Reduce salt and oxalate-rich foods (like spinach).
            - Follow up with a urologist.
        """)

    elif diagnosis == "Cyst":
        st.warning("**Kidney Cyst Detected** 💧")
        st.markdown("""
        - **Description**: A fluid-filled sac that can form in kidneys. Usually benign but requires monitoring.
        - **Symptoms**: Often asymptomatic. Large cysts may cause discomfort or high blood pressure.
        - **Next Steps**: Ultrasound or CT scan to monitor size and changes.
        - **Advice**:
            - Periodic checkups are important.
            - Maintain healthy kidney function (stay hydrated, avoid nephrotoxic drugs).
        """)
    
    elif diagnosis == "Normal":
        st.success("✅ No abnormality detected in this scan.")
        st.markdown("""
        - **Great news!** This scan appears normal.
        - **Advice**:
            - Keep a healthy lifestyle.
            - Drink plenty of water and have regular medical checkups.
            - If symptoms exist despite normal scan, consult a doctor for further evaluation.
        """)


def main():
    st.set_page_config(page_title="Kidney Scan Analyzer", layout="centered")

    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🩺 Kidney CT Scan Analysis System</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.write("Upload a CT scan image to classify and detect abnormalities such as **Tumor** and **Stone**.")

    uploaded_file = st.file_uploader("📤 Upload medical scan (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        classification_result, original_image = preprocess_image(uploaded_file)

        st.markdown("### 🖼 Uploaded Image & 🧠 Classification Result")

        col1, col2 = st.columns([1.5, 1])  # Wider image column

        with col1:
            st.image(original_image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown("#### 🧠 Classification Result")
            if classification_result == "Normal":
                st.success("✅ Normal scan detected.")
            elif classification_result in ["Stone", "Tumor", "Cyst"]:
                st.error(f"⚠️ Abnormal scan detected: **{classification_result}**")
            else:
                st.warning("Unknown classification result.")
            
        show_medical_info(classification_result)


        if classification_result in ["Stone", "Tumor"]:
            st.markdown("### 🔍 Detection Step")
            if st.button(f"Detect {classification_result}", type="primary"):
                with st.spinner("Running YOLO detection model..."):
                    detected_image, detection_results = detect_disease(original_image)

                    st.markdown("## 📌 Detection Results")
                    tab1, tab2 = st.tabs(["🖼 Visualization", "📋 Detection Info"])

                    with tab1:
                        st.image(detected_image, caption="Detection Output", use_container_width=True,width=500)

                    with tab2:
                        if len(detection_results.boxes) > 0:
                            st.markdown("### 🟢 Detected Regions:")
                            for box in detection_results.boxes:
                                class_id = int(box.cls)
                                confidence = float(box.conf)
                                st.info(f"🩺 **{model.names[class_id]}** detected with **{confidence:.2f}** confidence")
                        else:
                            st.warning("No abnormalities detected by the model.")
    else:
        st.info("Please upload a CT scan image to begin analysis.")

if __name__ == "__main__":
    main()
