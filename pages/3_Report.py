import streamlit as st
from Home import face_rec
import pandas as pd
# st.set_page_config(page_title='Reporting', layout='wide')
st.subheader('Reporting')

# Retrive Logs Data ANd Show in rporting
# extract data from redis list
name = 'attendance_logs'
def load_logs(name, end=-1):
    logs_list = face_rec.r.lrange(name,start= 0, end = end)
    return logs_list


# tabs to show the info
tab1, tab2 , tab3= st.tabs(['Registered Data','Logs', 'Attendance Report'])

with tab1:


    if st.button('Refresh Data'):
        # Retrive the data from Database
        with st.spinner('Retriving Data From Redis...'):

            redis_face_db = face_rec.retrive_data(name='academy_register')
            st.dataframe(redis_face_db[['Name','Role']])

with tab2:
    if st.button('Refresh Logs'):
        st.write(load_logs(name=name))
    
with tab3:
    st.subheader('Attendance Report')
    #load logs into attribute list

    log_list = load_logs(name=name)

    # Step 1: Convert the log
    convert_bytes_to_string = lambda x: x.decode('utf-8')
    log_list_string = list(map(convert_bytes_to_string,log_list))

    # st.write(log_list_string)
    # Step 2: Split the string by @ and create nested list 

    split_string = lambda x: x.split('@')
    log_nested_list = list(map(split_string,log_list_string))

    log_df = pd.DataFrame(log_nested_list, columns= ['Name', 'Role', 'Timestamp'])
    # st.write(log_df)

# Step 3: Time based report analysis
    
    log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
    log_df['Date'] = log_df['Timestamp'].dt.date



    report_df = log_df.groupby(by=['Date','Name','Role']).agg(
        In_time = pd.NamedAgg('Timestamp','min'),
        Out_time = pd.NamedAgg('Timestamp','max')
    ).reset_index()

    report_df['In_time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out_time'] = pd.to_datetime(report_df['Out_time'])

    report_df['Duration'] = report_df['Out_time'] - report_df['In_time']

    st.dataframe(report_df)
