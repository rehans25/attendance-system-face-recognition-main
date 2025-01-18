import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av 
from Home import face_rec



# st.set_page_config(page_title='Registration Form',)
st.subheader('Registration Form')

## init registration form
registrtion_form = face_rec.RegistrationForm()



#step 1: Collect person name and role
#form
person_name = st.text_input(label='Name', placeholder='Full Name')
role = st.selectbox(label='Select Your Role', options=('Student', 'Teacher'))

# Step 2: collect facial imbeddings of the person
def video_Frame_callback_fun(frame):

    img = frame.to_ndarray(format='bgr24') # 3d Array numpy array
    reg_img , embedding = registrtion_form.get_embedding(img)

    # save the data into local machine
    if embedding is not None:
        with open('face_embedding.txt', mode='ab') as f:
            np.savetxt(f,embedding)
   
    return av.VideoFrame.from_ndarray(reg_img, format='bgr24')


webrtc_streamer(key='registration', video_frame_callback=video_Frame_callback_fun,media_stream_constraints={"video": True, "audio": False},rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })


# Step 3: Save the Data in redis database

if st.button('Submit'):
    return_val = registrtion_form.save_data_redis(person_name, role)
    if return_val == True:
        st.success(f'{person_name} as {role} registered successfully')
    elif return_val == ('invalid_name'):
        st.error('Please enter valid name: Field cannot be empty or white spaces')
    elif return_val == ('file_not_found'):
        st.error(' "face_embedding.txt" file not found. Please refresh and execute again')
    


