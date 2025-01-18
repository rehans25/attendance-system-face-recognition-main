import numpy as np
import pandas as pd
import cv2
import redis
import os

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise 
import time
from datetime import datetime
# Connect To the Redis Client
# pass- AhrOKD0fBsLZW5KYxUK5hIwaRsjWtLFl
# host- redis-11969.c11.us-east-1-3.ec2.redns.redis-cloud.com
# port- 11969
hostname = 'redis-17692.c11.us-east-1-2.ec2.redns.redis-cloud.com'
port = 17692
password = "APm1hWOj0fEIoCiEsUt9CtlWBY6jg7M4"
r = redis.StrictRedis(
                host=hostname,
                port=port,
                password=password)

# Retrive the data from Database
def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role', 'facial_features']
    retrive_df[['Name', 'Role']] = retrive_df['name_role'].apply (lambda x: x.split ('@')).apply(pd.Series)
    return retrive_df[['Name','Role','facial_features']]

# Configure The FaceAnalysis

faceapp = FaceAnalysis (name='buffalo_sc', 
                      root='insightface_model',
                      providers= ['AzureExecutionProvider'] 
                     )
faceapp.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.5) # det_thresh always should be greater than 0.3

# Machine LEarning Search Algorithm


def ml_search_algo(dataframe,feature_column,test_vector,name_role=['Name','Role'],thresh=0.5):
    # Step -1: Take the dataframe (collection of data)
    dataframe = dataframe.copy()
    # Step-2: Index face embedding from the dataframe and convert into array
    x_list = dataframe[feature_column].tolist()
    x = np.asarray(x_list)
    # Step-3: Cal. cosine similarity
    similarity = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr = np.array(similarity).flatten()
    dataframe['cosine']= similar_arr
    # Step-4: filter the data
    data_filter = dataframe.query(f'cosine>= {thresh}')
    if len(data_filter) >0:
        # Step-5: Get the person name
        data_filter.reset_index(drop= True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'N/A'
    return person_name, person_role

## Realtime prediction
# we need to save log for every 10 min
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], role=[],current_time=[])
    def reset_dict(self):
        self.logs = dict(name=[], role=[],current_time=[])
    
    def saveLogs_redis(self):
        #step 1: create a logs data frame
        dataframe = pd.DataFrame(self.logs)

        #step 2: drop the duplicate info
        dataframe.drop_duplicates('name', inplace=True)
        # Step 3: Push data to Redis
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)
        if len(encoded_data) > 0:
            r.lpush('attendance_logs', *encoded_data)

        self.reset_dict()



    def face_prediction(self,test_image, dataframe,feature_column,name_role=['Name','Role'],thresh=0.5):
        # step 0: Find the time
        current_time = str(datetime.now())


        # Step 1: Apply the test image to  insightface
        results = faceapp.get(test_image)
        test_copy = test_image.copy()
        
        # Step 2: Use for Loop and extract each embeddings of test image and pass to ml_search_algo
        
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algo(dataframe,feature_column, test_vector=embeddings, name_role=name_role, thresh=thresh )
        
            if person_name == 'Unknown':
                color = (0,0,255)
            else:
                color = (0,255,0)
        
            cv2.rectangle(test_copy, (x1,y1),(x2,y2), color)
        
            text_gen = person_name
            cv2.putText(test_copy,text_gen,(x1,y1-15), cv2.QT_FONT_NORMAL, 0.5, color, 1)
            cv2.putText(test_copy,current_time,(x1,y2+15),cv2.QT_FONT_NORMAL, 0.5, color, 1)
            # save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)
        return test_copy
        
# Registration Form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0


    def get_embedding(self,frame):

        # get results from insightface model
        results = faceapp.get(frame,max_num=1) # max_num = 1 means only 1 person can be detected
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            # Put text for sample count
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1-10), cv2.QT_FONT_NORMAL, 0.7,(255,255,0), 1)
            
            # facial Features
            embeddings = res['embedding']
        return frame, embeddings

    def save_data_redis(self, name, role):

        #name validation
        if name is not None:
            if name.strip() != '':
                 key = f'{name}@{role}'
            else:
                return 'invalid_name'
        else:
            return 'invalid_name'

        # if face_embedding.txt exist
        if 'face_embedding.txt' not in os.listdir():
            return 'file_not_found'


        #step 1: load file"face_embedding.txt"
        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32) # flatten array

        #step 2: convert into array (proper shape)
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples,512)
        x_array = np.asarray(x_array)

        #step 3: cal. mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        #step 4: save this into redis database
        r.hset(name='academy_register', key=key,value=x_mean_bytes)


        os.remove('face_embedding.txt')
        self.reset()

        return True















