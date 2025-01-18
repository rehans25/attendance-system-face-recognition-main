import streamlit as st
# st.set_page_config(page_title='Attandance System', layout='wide')
st.header('Attendance System Using Face Recognition')

with st.spinner("Loading Models and Connecting to the Redis Client..."):
    import face_rec
st.success('Model loaded Successfully')
st.success('Redis Client successfully Connected')