import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model
model_gender = load_model('/home/subodh/without_mask_training/model-gender-dn-0.65.h5')
df_train = pd.read_csv('/home/subodh/Put_Mask/age_n_gen/training_annotations.csv')
train_dict = pd.Series(df_train.gender.values,index = df_train.path)
train_path_list = train_dict.keys()

def predict_gender(path):
    gender = -1
    try:
        img = cv2.imread('/home/subodh/without_mask_training/'+path)
        img = np.array(img)
        img = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
        img = img/255.0
        img = img -0.5
        img = img*2.0
        #print(img)
        img = img.reshape((1,128,128,3))
        predictions_gender = model_gender.predict(img)
        gender = np.argmax(predictions_gender[0])
    except:
        pass
    return gender

curated_gender_dict = {}

for i in range(len(train_path_list)-1):
    print(i)
    p = train_path_list[i]
    pg = predict_gender(p)
    print(pg)
    tg = train_dict[p]
    if (pg == tg):
        curated_gender_dict[p] = tg
        print('After removing miscalssified data the length of curated dictionary '+str(len(curated_gender_dict)))
df_curated = pd.DataFrame.from_dict(curated_gender_dict)
df_curated.to_csv('curated.csv')    
    