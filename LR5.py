# Step 1 & 2: Import libraries and configure Streamlit page
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import torch.nn.functional as F
import plotly.express as px 

# Step 1: Configure Streamlit page
st.set_page_config(page_title="CPU-based CV", layout="centered")
st.title("Image Classification with CPU-based Computer Vision")

st.markdown("""
This simple web application uses a **pre-trained ResNet18 convolutional neural network** to classify uploaded images.  
It runs entirely on **CPU**, making it accessible without specialized hardware.  

The app preprocesses the uploaded image, predicts the **top-5 classes**, and displays confidence probabilities in a **colorful bar chart** for easy understanding.
""")

# Step 3: Ensure CPU usage
device = torch.device("cpu")

# Step 4: Load pre-trained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(device)

# Step 5: Define image preprocessing transformations
preprocess = models.ResNet18_Weights.DEFAULT.transforms()

# Step 6: Upload image via Streamlit
uploaded_file = st.file_uploader("Please choose any image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Step 7: Preprocess image and convert to tensor
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Step 7 & 8: Model inference
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)

    # Get top-5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    categories = models.ResNet18_Weights.DEFAULT.meta["categories"]

    # Create DataFrame for top-5 predictions
    pred_data = []
    for i in range(top5_prob.size(0)):
        pred_data.append({
            "Class": categories[top5_catid[i]],
            "Probability (%)": float(top5_prob[i]*100)
        })
    df = pd.DataFrame(pred_data)

    # Display top-5 predictions
    st.subheader("🔹 Top-5 Predictions")
    for i, row in df.iterrows():
        st.write(f"{i+1}. {row['Class']} : {row['Probability (%)']:.2f}%")

    # Step 9: Bar chart
    st.subheader("📊 Top-5 Prediction Probabilities") 
    df = pd.DataFrame(pred_data)
    fig = px.bar(
        df,
        x='Class',
        y='Probability (%)',
        labels={'Probability (%)': 'Probability (%)', 'Class': 'Class'},
        color='Probability (%)',
        color_continuous_scale=["#844072", "#F50093", "#D9A0D5", "#FA63D2", "#76005B", "#E1BCD9"] 
    )
    st.plotly_chart(fig)