import mne
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
from Seizure_times import *
import numpy as np
from sklearn.model_selection import train_test_split
import collections
import os

CPS_seizures = []
elec_seizures = []
visual_seizures = []
noc_seizures = []
normals = []
labels = []
index_vect = []

for patient in [15, 14, 13, 12, 11, 10]:
    
    if patient == 10:
        files_list=["Record1.edf","Record2.edf"]
    elif patient in (15, 13, 11):
        files_list=["Record1.edf","Record2.edf","Record3.edf","Record4.edf"]
    elif patient in (14, 12):
        files_list=["Record1.edf","Record2.edf","Record3.edf"]
        
 
    for file_id, file in enumerate(files_list):
        file=os.path.join("./edfs", "p"+str(patient)+"_"+file)        
        data = mne.io.read_raw_edf(file, preload=True)               

        seizure_record = []
        index_mat = np.array([0,0])
        ###################### Ordering channels ######################
        if patient in (15, 14, 13, 12, 11):
            data.drop_channels(['EEG Cz-Ref', 'EEG Pz-Ref', 'ECG EKG', 'Manual']) # see attached word file for details
        elif patient == 10:
            data.reorder_channels(['EEG Fp2-Ref', 'EEG Fp1-Ref', 'EEG F8-Ref', 'EEG F4-Ref', 'EEG Fz-Ref', 'EEG F3-Ref', 'EEG F7-Ref', 'EEG A2-Ref', 'EEG T4-Ref', 'EEG C4-Ref', 'EEG C3-Ref', 'EEG T3-Ref', 'EEG A1-Ref', 'EEG T6-Ref', 'EEG P4-Ref', 'EEG P3-Ref', 'EEG T5-Ref', 'EEG O2-Ref', 'EEG O1-Ref'])
        print('#'*150)
        print('#'*150)
        print('#'*150)
        #print(data.info["ch_names"])
        print('#'*150)
        print('#'*150)
        print('#'*150)        
        raw_data = data.get_data()      # ndarray 19 x ~5M
        raw_data = np.array(raw_data)                   
        print(raw_data.shape)
        print('#'*150)
        print('#'*150)
        print('#'*150)  
        ###################################################
        record_time = data.info['meas_date']
        record_time = record_time.time()  #File time onset 
        record_time = datetime.combine(date.today(), record_time)
        # get seizure time and duration
        
        r = 'seizures_' + str(patient)
        for i in  range(len(eval(r)[file_id+1])):  #file_id correspond to record number so here we are checking how many
            # seizures inside a specific record
            s_time = time(eval(r)[file_id+1][i][0], eval(r)[file_id+1][i][1], eval(r)[file_id+1][i][2])
            seizure_duration = eval(r)[file_id+1][i][3]

            print('*'*150)
            print("File ID: {}, Record time:{}, Seizure time: {}, Seizure duration: {}".format('patient_' + str(patient) + '_Record'+str(file_id+1)+'_sz'+str(i+1), record_time, s_time, seizure_duration))
            print('*'*150)
            print('*'*150)
            print('*'*150)
            diff = datetime.combine(date.today(), s_time) - record_time # assigned date does not matter
            s_index = int( diff.total_seconds() * 500 ) # 500 sampling rate, 
            #s_index: seizure start in samples
            s_index_end = s_index + int(seizure_duration * 500)
            #s_index_end: seizure end in sample
            point_duration = 1  # how many seconds is one point
            #print(diff.days)
            #print(diff.seconds)
            #print(s_index)
            #print(s_index_end)
            print('*'*150)
            print('*'*150)
            print('*'*150)
            
            # get seizure index
             
            st = raw_data[:, s_index:s_index_end]
            
            index_mat = np.vstack([index_mat,[s_index,s_index_end]])  
            
            print(index_mat)                        

            if seizure_record == []:
                seizure_record = st
            else:
                seizure_record = np.concatenate((seizure_record,st),axis=1)
                #for patient 10, record2, the seizure is 305 sec, 305*500 = 152500 
           
            if len(index_mat) == 2:
                normal_record = np.delete(raw_data, np.s_[index_mat[1,0]:index_mat[1,1]], axis=1) # remove all data corresponding to a seizure    
               
                     
            elif len(index_mat) == 3:
                x1 = np.linspace(index_mat[1,0], index_mat[1,1], index_mat[1,1]-index_mat[1,0]+1, endpoint=True)
                x2 = np.linspace(index_mat[2,0], index_mat[2,1], index_mat[2,1]-index_mat[2,0]+1, endpoint=True)
                y = np.concatenate((x1,x2))
                normal_record = np.delete(raw_data, np.s_[y], axis=1)
        
            elif len(index_mat) == 4:
                x1 = np.linspace(index_mat[1,0], index_mat[1,1], index_mat[1,1]-index_mat[1,0]+1, endpoint=True)
                x2 = np.linspace(index_mat[2,0], index_mat[2,1], index_mat[2,1]-index_mat[2,0]+1, endpoint=True)
                x3 = np.linspace(index_mat[3,0], index_mat[3,1], index_mat[3,1]-index_mat[2,0]+1, endpoint=True)
                y = np.concatenate((x1,x2,x3))
                normal_record = np.delete(raw_data, np.s_[y], axis=1)
          
            elif len(index_mat) == 5:
                x1 = np.linspace(index_mat[1,0], index_mat[1,1], index_mat[1,1]-index_mat[1,0]+1, endpoint=True)
                x2 = np.linspace(index_mat[2,0], index_mat[2,1], index_mat[2,1]-index_mat[2,0]+1, endpoint=True)
                x3 = np.linspace(index_mat[3,0], index_mat[3,1], index_mat[3,1]-index_mat[2,0]+1, endpoint=True)
                x4 = np.linspace(index_mat[4,0], index_mat[4,1], index_mat[4,1]-index_mat[2,0]+1, endpoint=True)
                y = np.concatenate((x1,x2,x3,x4))
                normal_record = np.delete(raw_data, np.s_[y], axis=1)              
               
            print(seizure_record.shape)
            print('*'*150)
            print('*'*150)
            print('*'*150)

            if patient == 10:
                for i in range(seizure_duration):
                    data_point = seizure_record[:, i * 500:(i + 1) * 500]  #this will be 19x500
                    elec_seizures.append(data_point)  # this will a list of for example 305 matrix of size 19x500
            elif patient ==13 and file_id<3:
                for i in range(seizure_duration):
                    data_point = seizure_record[:, i * 500:(i + 1) * 500]  #this will be 19x500
                    noc_seizures.append(data_point)  # this will a list of for example 305 matrix of size 19x500            
            else:
                for i in range(seizure_duration):           # i is the one second interval 
                    data_point = seizure_record[:, i*500:(i+1)*500]
                    CPS_seizures.append(data_point)         # CPS is set of points, each of which is 19x500
    
            # normals
            for i in range(seizure_duration): #take same number of normal points
                data_point = normal_record[:, i*500:(i+1)*500]
                normals.append(data_point)                    


# transpose second and third dim
CPS_seizures = np.array(CPS_seizures)
scaler = np.amax(abs(CPS_seizures))
CPS_seizures = CPS_seizures/scaler

elec_seizures = np.array(elec_seizures)
scaler = np.amax(abs(elec_seizures))
elec_seizures = elec_seizures/scaler

noc_seizures = np.array(noc_seizures)
scaler = np.amax(abs(noc_seizures))
noc_seizures = noc_seizures/scaler

normals = np.array(normals)
scaler = np.amax(abs(normals))
normals = normals/scaler

# plt.figure('Normal')
# plt.plot(normals[100].T)
# plt.figure('Seizure')
# plt.plot(seizures[100].T)
# plt.show()
#

x = np.vstack((noc_seizures, elec_seizures, CPS_seizures, normals)) #trying to construct the x #points x 19 x 500

# below order matters
labels.extend([3 for i in range(len(noc_seizures))])
labels.extend([2 for i in range(len(elec_seizures))])
labels.extend([1 for i in range(len(CPS_seizures))])
labels.extend([0 for i in range(len(normals))])

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.1, random_state=1)

np.save("./data_aligned/x_train", x_train)
np.save("./data_aligned/x_test", x_test)
np.save("./data_aligned/y_train", y_train)
np.save("./data_aligned/y_test", y_test)

##############################################################################
import numpy as np
import scipy.io
matrix = np.load('./data_aligned/x_test.npy')
scipy.io.savemat('./data_aligned/x_test.mat', dict(x_test=matrix))
##############################################################################
matrix2 = np.load('./data_aligned/x_train.npy')
scipy.io.savemat('./data_aligned/x_train.mat', dict(x_train=matrix2))
##############################################################################
matrixy = np.load('./data_aligned/y_test.npy')
scipy.io.savemat('./data_aligned/y_test.mat', dict(y_test=matrixy))
##############################################################################
matrixy2 = np.load('./data_aligned/y_train.npy')
scipy.io.savemat('./data_aligned/y_train.mat', dict(y_train=matrixy2))