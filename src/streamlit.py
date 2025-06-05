import streamlit as st
from Preprocessing import preprocess_image ,load_model,detect_disease

model = load_model()
def main():
    st.title("Medical Image Analysis System")
    
    # Upload image
    uploaded_file = st.file_uploader("Upload medical scan", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # First step: Classify the image
        classification_result, original_image = preprocess_image(uploaded_file)
        
        # Display classification results
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Uploaded Image", width=300)
        with col2:
            st.write(f"Classification Result is : {classification_result}")
            if classification_result in ["Stone",'Tumor',"Cyst"]:
                st.error("Abnormal scan detected!")
            else:
                st.success("Normal scan detected")
        
        # Second step: Show detection button if classification is abnormal
        if classification_result in ["Stone","Tumor"]:
            if st.button(f"🔍 Detect the {classification_result} ", type="primary"):
                with st.spinner("Analyzing for specific conditions..."):
                    # Run detection
                    detected_image, detection_results = detect_disease(original_image)
                    
                    # Display results
                    st.subheader("Disease Detection Results")
                    
                    tab1, tab2 = st.tabs(["Visualization", "Details"])
                    
                    with tab1:
                        st.image(detected_image, caption="Detected Conditions", width=500)
                    
                    with tab2:
                        if len(detection_results.boxes) > 0:
                            st.write("### Detected Conditions:")
                            for box in detection_results.boxes:
                                class_id = int(box.cls)
                                confidence = float(box.conf)
                                st.write(f"- {model.names[class_id]} (confidence: {confidence:.2f})")
                        else:
                            st.info("No specific conditions detected")

if __name__ == "__main__":
    main()