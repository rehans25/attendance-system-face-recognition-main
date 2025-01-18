import streamlit as st
# st.set_page_config(page_title='Real Time Prediction')
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time
import cv2
st.subheader('Real-Time Prediction')

# Retrive the data from Database
with st.spinner('Retriving Data From Redis...'):

    redis_face_db = face_rec.retrive_data(name='academy_register')
    st.dataframe(redis_face_db)
st.success('Data successfully retrived from Redis')

#time
waitTime = 5 # time in seconds in which log will be gener
setTime = time.time()
realtimepred = face_rec.RealTimePred()


# Realtime Prediction
# Streamlit Webrtc

 # callback function
def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24") # 3d Array numpy array
    #operations that you can perform
    pred_img= realtimepred.face_prediction(img,redis_face_db,'facial_features',['Name', 'Role'], thresh=0.5)

    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time() # reset Time

        print('Save data to Redis Database')

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key='realtimePrediction', video_frame_callback=video_frame_callback,media_stream_constraints={"video": True, "audio": False},rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })







