
# import all required libaries
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from PIL import Image
import requests
from io import BytesIO
st.set_option('deprecation.showfileUploaderEncoding', False)

response = requests.get(url='https://api.github.com/favicon.ico')
im = Image.open(BytesIO(response.content))

st.set_page_config(
    page_title='Image FaceMask Detection App' ,
    page_icon = im , 
    layout = 'wide' , 
    initial_sidebar_state = 'auto'
)

st.title("Image FaceMask Detection App :mask:")
st.write("This App detects **the person is wearing face mask or not by uploading face Image!**")
st.write('---')

st.sidebar.image('Image-Face-Mask-Detection-App.png')
st.sidebar.title('Image FaceMask Detection App :mask:')
st.sidebar.write('---')

# st.cache allow our app to stay performant (run fast) since
# we will be loading images
@st.cache(hash_funcs={cv2.dnn_Net: hash})
# this is a helper function to load our face detector model
# from desk
def load_face_detector_model():
    prototxt_path = os.path.sep.join(
        ["face_detector", "deploy.prototxt"])
    weight_path = os.path.sep.join(
        ['face_detector', 'res10_300x300_ssd_iter_140000.caffemodel'])

    net = cv2.dnn.readNet(prototxt_path, weight_path)

    return net


# This will make the app stay performant
@st.cache(allow_output_mutation=True)
# helper function to load our face mask detector model
def load_mask_model():

    mask_model = load_model("mask_detector.model")

    return mask_model


net = load_face_detector_model() # load face detector model
model = load_mask_model() # load mask detector model
confidence_selected = st.sidebar.slider(
    'Select a confidence range', 0.0, 0.1, 0.5, 0.1) # display button to adjust 'confidence' between 0 - 0.5

# Helper function to load the image and loop over the detection
def detect_mask(image):

    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)  #read the image from tempoary memory
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert image from BGR to RGB
    orig = image.copy() # get a copy of the image
    (h, w) = image.shape[:2] # get image height and weight
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), # construct a blob from the image
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)  # pass the blob through the detection, get region that differ in propertes, and the face region
    detection = net.forward() 

    for i in range(0, detection.shape[2]): # loop through the detection

        confidence = detection[0, 0, i, 2] # extract confidence vaalue

        if confidence > confidence_selected: # if the confidence is greater than the selected confidence from the side bar

            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h]) # get x and y coordinate for the bounding box
            (startX, startY, endX, endY) = box.astype("int") 

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY)) # ensure bounding box does not exceed image frame

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # extract face ROI, convert from BGR to RGB
            face = cv2.resize(face, (224, 224))         # resize to 224, 224
            face = img_to_array(face)                      # convert resized face to an array
            face = preprocess_input(face)               # preprocess the array
            face = np.expand_dims(face, axis=0)            # expand array to 2D

            (mask, withoutMask) = model.predict(face)[0] # pass the face through the mask model, detect if there is mask or not

            label = "Mask" if mask > withoutMask else "No Mask" # define label
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # bbox is Green if 'mask' else Blue

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100) # add label probability 

            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2) #display label and bbox rectangle in output frame

        return image, label # return image and label

# Helper function for the main page

def main():
    
    # activities = ['Home']
    # choice = st.sidebar.selectbox("Hey, What do want to do", activities) # streammlit function to display box selector
    # image, label = detect_mask(image=image)

    # if choice == "Home": # If user chooses Home page
        image_file = st.file_uploader("Upload image", type=['jpeg', 'jpg', 'png']) # streamlit function to upload file
        

        if image_file is not None:  #confirm that the image is not a 0 byte file


            st.sidebar.image(image_file, width=240) # then display a sidebar of the uploaded image
            
            if st.button("Process"): # process button
            
                image, label = detect_mask(image_file) # call mask detection model
                st.image(image, width=420) # display the uploaded image
                st.info('### ' +  label) # dispaly label



if __name__ == "__main__": 
    main()

hide_streamlit_style = '''
            <style>
            footer {visibility: hidden;}
            </style>
            '''
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.write("This App is built by **Ishan Chakraborty**.")