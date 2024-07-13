import tensorflow as tf
import pandas as pd
import numpy as np
#import keras.backend as k
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold,KFold,GroupKFold
from sklearn.cluster import KMeans
import plotly.express as px
import tensorflow.keras.layers as L

def allocate_gpu_memory(gpu_number=0):
    #Lists all gpu available on the machine
    physical_devices=tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            print("Found {} GPU".format(len(physical_devices)))
            #Set the visible device to the specified GPU
            tf.config.set_visible_devices(physical_devices[gpu_number],'GPU')
            #Enabling memory growth on specified GPU
            tf.config.experimental.set_memory_growth(physical_devices[gpu_number],True)
            print("#{} GPU memory is allocated".format(gpu_number))
        except RuntimeError as e:
            print(e)
    else:
        print("Not enough GPU hardware devices available")
        
#Allocate GPU memory for GPU 0
allocate_gpu_memory()

#varibale storing the version name of the model configaration
Ver='GRU_LSTM1'
#debug flag for logging or other debug related purposes
debug = False
#Path to augmented data for input
#aug_data='../input/openvaccine-augmentation-data/aug_data1.csv'

def gru_layer(hidden_dim,dropout):
    return L.Bidirectional(L.GRU(hidden_dim,dropout=dropout,return_sequences=True,kernel_initialiser='orthogonal'))
    #L.Bidrectional -> Makes Gated recurrent unit to process input sequences in both forward and backward direction
    #kernel_initialiser= initialised the weight using orthogonal initialisation method
    #L.GRU is for initialising GRU layer itself
    #dropout rate for regularisation method is rate at which multicollinearity and  overfitting is corrected

#LSTM-Long short term memory function
def lstm_layer(hidden_dim,dropout):
    return L.Bidirectional(L.LSTM(hidden_dim,dropout=dropout,return_sequence=True,kernel_initialiser='orthogonal'))

def build_model(seq_len=107,pred_len=68,dropout=0.5,embed_dim=100,hidden_dim=256,type=0):
    inputs=L.Input(shape=(seq_len,6))
    categorical_feat_dim=3
    categorical_fea=inputs[:,:,:categorical_feat_dim]
    numerical_fea=input[:,:,3:]
    #Convert categorical features into dense vectors
    embed=L.Embedding(input_dim=len(token2int),output_dim=embed_dim)(categorical_fea)
    #Reshape embed features to concatenate with numerical features
    reshaped=tf.reshape(embed,shape=(-1,embed.shape[1],embed.shape[2]*embed.shape[3]))
    #Concatenate the reshaped embedings with numerical features along the last axis
    reshaped=L.concatenate([reshaped,numerical_fea],axis=2)
    #Based on the 'type' parameter the recurrent neural network layers are created and input tensor is applied sequentially
    if type==0:
        hidden=gru_layer(hidden_dim,dropout)(reshaped)
        hidden=gru_layer(hidden_dim,dropout)(hidden)
    elif type == 1:
        hidden=lstm_layer(hidden_dim,dropout)(reshaped)
        hidden=gru_layer(hidden_dim,dropout)(hidden)
    elif type == 2:
        hidden=gru_layer(hidden_dim,dropout)(reshaped)
        hidden=lstm_layer(hidden_dim,dropout)(hidden)
    elif type==3:
        hidden=lstm_layer(hidden_dim,dropout)(reshaped)
        hidden=lstm_layer(hidden_dim,dropout)(hidden)
    #Truncating the ouput of the recurrent layers to pred_len lenght(length of the output sequence i.e around 68)
    truncated=hidden[:,:pred_len]
    out=L.Dense(5,activation='Linear')(truncated)
    model=tf.keras.Model(inputs=inputs,outputs=out)
    model.compile(tf.keras.optimizers.Adam(),loss=mcrmse)
    return model

token2int={x:i for i,x in enumerate('().ACGUBEHIMSX')}
pred_cols=['reactivity','deg_Mg_pH10','deg_Mg_50C','deg_50C']

def preprocess_inputs(df,cols=['sequence','structure','predicted_loop_type']):
    '''Three columns are processed. 
    each column is a sequence and is coverted into numeric by preprocessing and later converted into list.
      All three features of data frame is a list of value of all the sequence.Transposing it changes the dimension
        such that 107 rows represent each item in sequence with 679 values in each list representing all the 679 train sample 
        for that partiular sequence for each row representing item of sequence and 3 columns representing three features.
        Transposing with (0,2,1) puts all 3 values of each element of a sequence to one array and it has 107 columns represeing all three values of all 107 items of each sample and total 629 rows reprsenting each sample. '''
    base_fea=np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq:[token2int[x] for x in seq])
            .values
            .tolist()
            ),
            (0,2,1)
    )
    bpps_sum_fea=np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]
    bpps_max_fea=np.array(df['bpps_max'].to_list())[:,:,np.newaxis]
    bpps_nb_fea=np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]
    return np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea,bpps_nb_fea],2)
def rmse(y_actual,y_pred):
    mse=tf.keras.losses.mean_squared_error(y_actual,y_pred)
    return tf.sqrt(mse)

def mcrmse(y_actual,y_pred,num_scored=len(pred_cols)):
    score = 0
    for i in range(num_scored):
        score+=rmse(y_actual[:,:,i],y_pred[:,:,i])/num_scored
    return score

#Load and preprocess the data
train=pd.read_json('/Users/pruthvikbr/Documents/Kaggle/stanford-covid-vaccine/train.json',lines=True)
test=pd.read_json('/Users/pruthvikbr/Documents/Kaggle/stanford-covid-vaccine/test.json',lines=True)

#Additional Preprocessing and feature formation

def read_bpps_sum(df):
    bpps_arr=[]
    for mol_id in df.id.to_list():
        #Sum of probabilities over each row signify the distrubution equals to 1
        bpps_arr.append(np.load(f'/Users/pruthvikbr/Documents/Kaggle/stanford-covid-vaccine/bpps/{mol_id}.npy').sum(axis=1))
    return bpps_arr

def read_bpps_max(df):
    bpps_arr=[]
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f'/Users/pruthvikbr/Documents/Kaggle/stanford-covid-vaccine/bpps/{mol_id}.npy').max(axis=1))
    return bpps_arr

def read_bpps_nb(df):
    #normalizing the value
    bpps_nb_mean=0.077522
    bpps_nb_std=0.08914
    bpps_arr=[]
    for mol_id in df.id.to_list():
        bpps=np.load(f'/Users/pruthvikbr/Documents/Kaggle/stanford-covid-vaccine/bpps/{mol_id}.npy')
        #Counts all the bases which has probability to pair for a particular base and dividies it by total number of bases. This gives percentage ratio of bases it can pair with
        bpps_nb=(bpps>0).sum(axis=0)/bpps.shape[0]
        #Mean value of ratio of number of bases a particular base can pair with - it's actual ratio value of a base / standar devaiton. Gives Z-Score considered as normalised value
        bpps_nb=(bpps_nb-bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr

#sample=np.load('/Users/pruthvikbr/Documents/Kaggle/stanford-covid-vaccine/bpps/id_0a2bbe37e.npy')
#(sample > 0).sum(axis=0) / sample.shape[0]
#sample_mean=0.077522
#sample_std=0.08914
#(((sample > 0).sum(axis=0) / sample.shape[0]) - sample_mean) / sample_std

#creation of new features for both Train and test
train['bpps_sum'] = read_bpps_sum(train)
test['bpps_sum'] = read_bpps_sum(test)
train['bpps_max'] = read_bpps_max(train)
test['bpps_max'] = read_bpps_max(test)
train['bpps_nb'] = read_bpps_nb(train)
test['bpps_nb'] = read_bpps_nb(test)


#Pre-processing the input data
# clustering for GroupKFold
# expecting more accurate CV by putting similar RNAs into the same fold.
kmeans_model = KMeans(n_clusters=200, random_state=110).fit(preprocess_inputs(train)[:,:,0])
train['cluster_id'] = kmeans_model.labels_


#Build and Train Model

model=build_model()
model.summary()

def train_and_predict(type=0,FOLD_N=5):
    #initialising GropuKFOLD model for cross  Validation
    grpkfld=GroupKFold(n_splits=FOLD_N)

    #segregating private and public inputs  by checking for sequence length
    public_df=test.query("seq_length==107").copy
    private_df=test.query("seq_length==130").copy

    #Preprocessing public and private df's
    public_inputs=preprocess_inputs(public_df)
    private_inputs=preprocess_inputs(private_df)

    #Store validation sets and their corresponding predictions
    holdouts=[]
    holdout_preds=[]

    #We want to loop through each fold created usin GroupKfold 
    for cv,(tr_idx,vl_idx) in enumerate(grpkfld.split(train,train['reactivity'],train['cluster_id'])):
        trn=train.iloc[tr_idx]
        x_trn=preprocess_inputs(trn)
        y_trn=np.array(trn[pred_cols].values.to_list()).transpose((0,2,1))
        w_trn=np.log(trn.signal_to_noise+1.1)/2

        #Validation data preparation
        val = train.iloc[vl_idx]
        x_val_all = preprocess_inputs(val)
        val = val[val.SN_filter == 1]
        x_val = preprocess_inputs(val)
        y_val = np.array(val[pred_cols].values.tolist()).transpose((0, 2, 1))

        #Model Building for general training,prediction on sequences of length 107, on length of 130
        model = build_model(type=type)
        model_short = build_model(seq_len=107, pred_len=107,type=type)
        model_long = build_model(seq_len=130, pred_len=130,type=type)
        #If the monitored metric does not improve for a certain number of epochs (patience), the learning rate is reduced by a specified factor.this refers to callback function in the below fit function
        history=model.fit(x_trn,y_trn,validation_data=(x_val,y_val),batch_size=64,epochs=60,sample_weight=w_trn,callbacks=[tf.keras.callbacks.ReduceLROnPlateau(),tf.keras.callbacks.ModelCheckpoint(f'model{Ver}_cv{cv}.h5')])

        #plotting the training history
        fig = px.line(history.history, y=['loss', 'val_loss'],labels={'index': 'epoch', 'value': 'Mean Squared Error'},title='Training History')
        fig.show()

        #Loading the best weights into the model
        model.load_weights(f'model{Ver}_cv{cv}.h5')
        model_short.load_weights(f'model{Ver}_cv{cv}.h5')
        model_long.load_weights(f'model{Ver}_cv{cv}.h5')

        #store Validation Predictions
        holdouts.append(train.iloc[vl_idx])
        holdout_preds.append(model.predict(x_val_all))
        #Making Test Predictions. making for public and private test. Dividing by fold_n to average the predictions over all the fold
        if cv == 0:
            public_preds = model_short.predict(public_inputs)/FOLD_N
            private_preds = model_long.predict(private_inputs)/FOLD_N
        else:
            public_preds += model_short.predict(public_inputs)/FOLD_N
            private_preds += model_long.predict(private_inputs)/FOLD_N
    #The function returns the validation sets, their predictions, and the test datasets with their predictions.
    return holdouts, holdout_preds, public_df, public_preds, private_df, private_preds    

val_df, val_preds, test_df, test_preds = [], [], [], []
if debug:
    nmodel = 1
else:
    nmodel = 4
for i in range(nmodel):
    holdouts, holdout_preds, public_df, public_preds, private_df, private_preds = train_and_predict(i)
    val_df += holdouts
    val_preds += holdout_preds
    test_df.append(public_df)
    test_df.append(private_df)
    test_preds.append(public_preds)
    test_preds.append(private_preds)







#POST PROCESS

preds_ls = []
for df, preds in zip(test_df, test_preds):
    for i, uid in enumerate(df.id):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
        preds_ls.append(single_df)
preds_df = pd.concat(preds_ls).groupby('id_seqpos').mean().reset_index()
# .mean() is for
# 1, Predictions from multiple models
# 2, TTA (augmented test data)

preds_ls = []
for df, preds in zip(val_df, val_preds):
    for i, uid in enumerate(df.id):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
        single_df['SN_filter'] = df[df['id'] == uid].SN_filter.values[0]
        preds_ls.append(single_df)
holdouts_df = pd.concat(preds_ls).groupby('id_seqpos').mean().reset_index()

submission = preds_df[['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]
submission.to_csv(f'submission.csv', index=False)
print(f'wrote to submission.csv')

#VALIDATION
def print_mse(prd):
    val = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

    val_data = []
    for mol_id in val['id'].unique():
        sample_data = val.loc[val['id'] == mol_id]
        sample_seq_length = sample_data.seq_length.values[0]
        for i in range(68):
            sample_dict = {
                           'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),
                           'reactivity_gt' : sample_data['reactivity'].values[0][i],
                           'deg_Mg_pH10_gt' : sample_data['deg_Mg_pH10'].values[0][i],
                           'deg_Mg_50C_gt' : sample_data['deg_Mg_50C'].values[0][i],
                           }
            val_data.append(sample_dict)
    val_data = pd.DataFrame(val_data)
    val_data = val_data.merge(prd, on='id_seqpos')

    rmses = []
    mses = []
    for col in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:
        rmse = ((val_data[col] - val_data[col+'_gt']) ** 2).mean() ** .5
        mse = ((val_data[col] - val_data[col+'_gt']) ** 2).mean()
        rmses.append(rmse)
        mses.append(mse)
        print(col, rmse, mse)
    print(np.mean(rmses), np.mean(mses))



print_mse(holdouts_df)
print_mse(holdouts_df[holdouts_df.SN_filter == 1])

 # type: ignore