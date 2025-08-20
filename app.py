# Simple web demo for the application process.
# Floralpina is a fictional company.

import streamlit as st
import pickle
from PIL import Image
from utils import predict_one, iter_top_proba_predictions, classes, load_model, CenterCropMin

# Title
st.title("Floralpina Demo App")

# Login with Google / Microsoft
if not st.user.is_logged_in:
    st.write("Please log in to continue.")

    col1, col2 = st.columns([1,3])

    with col1:
        st.button("Log in with Google", on_click=st.login, args=["google"])

    with col2:
        st.button("Log in with Microsoft", on_click=st.login, args=["microsoft"])
    st.stop()

# Ensure user is allowed access
if st.user.email not in st.secrets['auth']['allowed_users']:
    st.error("You are not authorized to use this app.")
    st.button("Log out", on_click=st.logout)
    st.stop()

# Logout
st.button("Log out", on_click=st.logout)
st.markdown(f"Welcome {st.user.name}!")

# Header
st.header("Get to know that flower! Upload a picture")

# Accept image upload
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"]
)

# Wait until image has been uploaded
if uploaded_file is None:
    st.stop()

# Image has been uploaded.

# Cache the model
@st.cache_resource
def get_model():
    return load_model('./data/model/last_layer.pt')

centercrop = CenterCropMin()

# Loading the model and getting predictions might take a while.
# Display a spinner while busy.
col1, col2 = st.columns([1,2])
with st.spinner("Making prediction..."):
    # Open image with PIL
    image = Image.open(uploaded_file)
    image = centercrop(image)

    # Display the image
    col1.image(image, caption="Uploaded Image", use_container_width=True)

    # Load the model
    model = get_model()

    # Get predictions
    p = predict_one(model, image)

    # Get top proba indices
    top_proba_idx = list(iter_top_proba_predictions(p, n_sigma=2))

    # Display results
    if len(top_proba_idx) > 0:
        col2.subheader("Classification results:")
        for i in top_proba_idx:
            col2.success(f"{classes[i].capitalize()} ({100*p[i]:.0f}% confident)")
    # No results found
    else:
        col2.error("Unable to classify this image right now. Please try again or try another image.")
