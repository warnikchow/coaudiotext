# CoAudioText
A short tutorial for the co-utilization of audio and text data (multi-modal analysis)

## Contents
[0. Problem definition & loading dataset](https://github.com/warnikchow/coaudiotext/blob/master/README.md#0-problem-definition--loading-dataset)</br>
[1. Extracting acoustic features](https://github.com/warnikchow/coaudiotext/blob/master/README.md#1-extracting-acoustic-features)</br>
[2. Speech-only analysis with Librosa and Keras](https://github.com/warnikchow/coaudiotext/blob/master/README.md#2-speech-only-analysis-with-librosa-and-keras)</br>
[3. Self-attentive BiLSTM](https://github.com/warnikchow/coaudiotext/blob/master/README.md#3-self-attentive-bilstm)</br>
[4. Parallel utilization of audio and text data](https://github.com/warnikchow/coaudiotext/blob/master/README.md#4-parallel-utilization-of-audio-and-text-data)</br>
[5. Multi-hop attention](https://github.com/warnikchow/coaudiotext/blob/master/README.md#5-multi-hop-attention)</br>
[6. Cross-attention](https://github.com/warnikchow/coaudiotext/blob/master/README.md#6-cross-attention)

## 0. Problem definition & loading dataset

Understanding the intention of an utterance is challenging for some prosody-sensitive cases, especially when it is in the written form as in a text chatting or speech recognition output. **The main concern is detecting the directivity or rhetoricalness of an utterance and distinguishing the type of question.** Since it is inevitable to face both the issues regarding prosody and semantics, the identification is expected be benefited from the observations on human language processing mechanism. 

Challenging issues for spoken language understanding (SLU) modules include inferring the intention of syntactically ambiguous utterances. If an utterance has an underspecified sentence ender whose role is decided only upon prosody, the inference requires whole the acoustic and textual data of the speech for SLUs (and even human) to correctly infer the intention, since the pitch sequence, the duration between the words, and the overall tone decides the intention of the utterance. For example, in Seoul Korean which is *wh-in-situ*, many sentences incorporate various ways of interpretation that depend on the intonation, as shown in our [ICPhS paper](http://www.assta.org/proceedings/ICPhS2019/papers/ICPhS_3951.pdf).

Here, we attack the issue above utilizing the speech corpus that is distributed along with the paper. The scripts and speech files are available in [this github repository](https://github.com/warnikchow/prosem). As you download the folder from [the dropbox](https://www.dropbox.com/s/3tm6ylu21jpmnj8/ProSem_KOR_speech.zip?dl=0), unzip the folder in YOUR DIRECTORY so that you have *ProSem_KOR_speech* folder there. In it, there are the folders named *FEMALE* and *MALE* each containing 3,551 Korean speech utterances. Also, place the folder *text* in YOUR DIRECTORY, which contains the scripts of the speech files.

## 1. Extracting acoustic features

First, since we only utilize the utterances that can be disambiguated with speech, here we extract the acoustic features from the files. There are many ways to abstractize the physical components, but here we utilize [mel spectrogram](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0) due to its rich acoustic information and intuitive concept. The process is done with [Librosa](https://librosa.github.io/librosa/), a python-based audio signal processing library.

```python
def make_data(fname,fnum,shuffle_name,mlen):
    data_s_rmse = np.zeros((fnum,mlen,129))
    for i in range(fnum):
        if i%200 ==0:
            print(i)
        num = str(shuffle_name[i])
        filename = fname+num+'.wav'
        y, sr = librosa.load(filename)
        D = np.abs(librosa.stft(y))**2
        ss, phase = librosa.magphase(librosa.stft(y))
        rmse = librosa.feature.rmse(S=ss)
        rmse = rmse/np.max(rmse)
        rmse = np.transpose(rmse)
        S = librosa.feature.melspectrogram(S=D)
        S = np.transpose(S)
        if len(S)>=mlen:
            data_s_rmse[i][:,0]=rmse[-mlen:,0]
            data_s_rmse[i][:,1:]=S[-mlen:,:]
        else:
            data_s_rmse[i][-len(S):,0]=np.transpose(rmse)
            data_s_rmse[i][-len(S):,1:]=S
    return data_s_rmse

fem_speech = make_data('speech/FEMALE/',3552,x_fem,200)
mal_speech = make_data('speech/MALE/',3552,x_mal,200)

total_speech_train = np.concatenate([fem_speech[:3196],mal_speech[:3196]])
total_speech_test  = np.concatenate([fem_speech[3196:],mal_speech[3196:]])
total_speech = np.concatenate([total_speech_train,total_speech_test])
```

## 2. Speech-only analysis with Librosa and Keras

Although people these days seem to migrate to TF2.0 and PyTorch, I still use [original Keras](https://keras.io/) for its conciseness. Hope this code be fit to whatever flatform the reader uses.

```python
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.layers import TimeDistributed, Bidirectional, Concatenate
from keras.layers import Input, Embedding, LSTM, GRU, SimpleRNN, Lambda
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import keras.layers as layers

from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)
```

The next step is required for a balanced training and evaluation. *class_weights* denote the ratio that is to be weighted in training phase regarding the corpus utterance type volume, and by defining *F1 score*, we can get the point of the evaluation which *accuracy* usually fails to discern while using the imbalanced corpus.

```python
##### class_weights
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(fem_label), fem_label)

##### f1 score ftn.
from keras.callbacks import Callback
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metricsf1macro(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_f1s_w = []
        self.val_recalls_w = []
        self.val_precisions_w = []
    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.asarray(self.model.predict(self.validation_data[0]))
        val_predict = np.argmax(val_predict,axis=1)
        val_targ = self.validation_data[1]
        _val_f1 = metrics.f1_score(val_targ, val_predict, average="macro")
        _val_f1_w = metrics.f1_score(val_targ, val_predict, average="weighted")
        _val_recall = metrics.recall_score(val_targ, val_predict, average="macro")
        _val_recall_w = metrics.recall_score(val_targ, val_predict, average="weighted")
        _val_precision = metrics.precision_score(val_targ, val_predict, average="macro")
        _val_precision_w = metrics.precision_score(val_targ, val_predict, average="weighted")
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_f1s_w.append(_val_f1_w)
        self.val_recalls_w.append(_val_recall_w)
        self.val_precisions_w.append(_val_precision_w)
        print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))
        print("— val_f1_w: %f — val_precision_w: %f — val_recall_w: %f"%(_val_f1_w, _val_precision_w, _val_recall_w))

metricsf1macro = Metricsf1macro()
```

The following denotes how we define the BiLSTM by using Keras, although no functional API is utilized here. We use only *Sequential()* since no more complex structure is used. We don't use dropout here since the hidden layers of this size are not expected to get overhead.

```python
def validate_bilstm(rnn_speech,train_y,hidden_lstm,hidden_dim,cw,val_sp,bat_size,filename):
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_lstm), input_shape=(len(rnn_speech[0]), len(rnn_speech[0][0]))))
    model.add(layers.Dense(hidden_dim, activation='relu'))
    model.add(layers.Dense(int(max(y)+1), activation='softmax'))
    model.summary()
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro,checkpoint]
    model.fit(rnn_speech,train_y,validation_split=val_sp,epochs=100,batch_size=bat_size,callbacks=callbacks_list,class_weight=cw)

validate_bilstm(total_speech,total_label,64,128,class_weights,0.1,16,'model_icassp/total_bilstm')
```

## 3. Self-attentive BiLSTM

Remember: mel spectrogram still has a plenty of prosody-semantic information! Thus, we decided to apply a self-attentive embedding which has been successfully used in text procesisng. Before making up the module, in terms of *pure Keras* where F1 measure is removed (well if recent version has one, that's nice!), we need another definition of f1 score since additional input source is introduced (zero vector for attention source initialization).

```python
class Metricsf1macro_2input(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_f1s_w = []
        self.val_recalls_w = []
        self.val_precisions_w = []
    def on_epoch_end(self, epoch, logs={}):
        if len(self.validation_data)>2:
            val_predict = np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1]]))
            val_predict = np.argmax(val_predict,axis=1)
            val_targ = self.validation_data[2]
        else:
            val_predict = np.asarray(self.model.predict(self.validation_data[0]))
            val_predict = np.argmax(val_predict,axis=1)
            val_targ = self.validation_data[1]
           _val_f1 = metrics.f1_score(val_targ, val_predict, average="macro")
           _val_f1_w = metrics.f1_score(val_targ, val_predict, average="weighted")
           _val_recall = metrics.recall_score(val_targ, val_predict, average="macro")
           _val_recall_w = metrics.recall_score(val_targ, val_predict, average="weighted")
           _val_precision = metrics.precision_score(val_targ, val_predict, average="macro")
           _val_precision_w = metrics.precision_score(val_targ, val_predict, average="weighted")
           self.val_f1s.append(_val_f1)
           self.val_recalls.append(_val_recall)
           self.val_precisions.append(_val_precision)
           self.val_f1s_w.append(_val_f1_w)
           self.val_recalls_w.append(_val_recall_w)
           self.val_precisions_w.append(_val_precision_w)
           print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))
           print("— val_f1_w: %f — val_precision_w: %f — val_recall_w: %f"%(_val_f1_w, _val_precision_w, _val_recall_w))

metricsf1macro_2input = Metricsf1macro_2input()
```

And here we define our self-attentive bilstm model which *sometimes* uses TensorFlow backend. This kind of design (utilizinig *Model* module) is inevitable since the pure Keras approach cannot guarantee that we can make up such a complicated layer... So, I attach a rather detailed comment to help the readers follow how the structure in [the paper](https://arxiv.org/abs/1703.03130) is implemented as a code.

```python
def validate_rnn_self_drop(rnn_speech,train_y,hidden_lstm,hidden_con,hidden_dim,cw,val_sp,bat_size,filename):
    ##### Speech input and BiLSTM with hidden layer sequence
    speech_input = Input(shape=(len(rnn_speech[0]),len(rnn_speech[0][0])),dtype='float32')
    speech_layer = Bidirectional(LSTM(hidden_lstm,return_sequences=True))(speech_input)
    ##### Single layer perceptron for making up a context vector (of size hidden_con)
    speech_att = Dense(hidden_con, activation='tanh')(speech_layer)
    ##### Zeros-sourced attention vector computation and SLP (to make size hidden_con)
    att_source   = np.zeros((len(rnn_speech),hidden_con))
    att_input    = Input(shape=(hidden_con,), dtype='float32')
    att_vec = Dense(hidden_con,activation='relu')(att_input)
    att_vec = Dropout(0.3)(att_vec)
    att_vec = Dense(hidden_con,activation='relu')(att_vec)
    ##### Attention vector as an outupt of column-wise dot product of att_vec and speech_att
    att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([att_vec,speech_att])
    att_vec = Dense(len(rnn_speech[0]),activation='softmax')(att_vec)
    att_vec = layers.Reshape((len(rnn_speech[0]),1))(att_vec)
    ##### Column-wise scalar  multiplication of attention weight and the hidden layer sequence
    speech_layer  = layers.multiply([att_vec,speech_layer])
    ##### Summation for a final output + Dropouts to prevent overhead
    speech_output = Lambda(lambda x: K.sum(x, axis=1))(speech_layer)
    speech_output = Dense(hidden_dim, activation='relu')(speech_output)
    speech_output = Dropout(0.3)(speech_output)
    speech_output = Dense(hidden_dim, activation='relu')(speech_output)
    speech_output = Dropout(0.3)(speech_output)
    ##### Final softmax layer
    main_output = Dense(int(max(train_y)+1),activation='softmax')(speech_output)
    model = Sequential()
    model = Model(inputs=[speech_input,att_input],outputs=[main_output])
    model.summary()
    model.compile(optimizer=adam_half,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro_2input,checkpoint]   
    model.fit([rnn_speech,att_source],train_y,validation_split=val_sp,epochs=100,batch_size=bat_size, callbacks=callbacks_list,class_weight=cw)

validate_rnn_self_drop(total_speech,total_label,64,64,128,class_weights,0.1,16,'model_icassp/total_bilstm_att')
```

## 4. Parallel utilization of audio and text data

```python

def validate_speech_self_text_self(rnn_speech,rnn_text,train_y,hidden_lstm_speech,hidden_con,hidden_lstm_text,hidden_dim,cw,val_sp,bat_size,filename):
    ##### Speech BiLSTM-SA
    speech_input = Input(shape=(len(rnn_speech[0]),len(rnn_speech[0][0])), dtype='float32')
    speech_layer = Bidirectional(LSTM(hidden_lstm_speech,return_sequences=True))(speech_input)
    speech_att   = Dense(hidden_con, activation='tanh')(speech_layer)
    speech_att_source= np.zeros((len(rnn_speech),hidden_con))
    speech_att_input = Input(shape=(hidden_con,),dtype='float32')
    speech_att_vec   = Dense(hidden_con, activation='relu')(speech_att_input)
    speech_att_vec   = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([speech_att_vec,speech_att])
    speech_att_vec   = Dense(len(rnn_speech[0]),activation='softmax')(speech_att_vec)
    speech_att_vec   = layers.Reshape((len(rnn_speech[0]),1))(speech_att_vec)
    speech_output= layers.multiply([speech_att_vec,speech_layer])
    speech_output= Lambda(lambda x: K.sum(x, axis=1))(speech_output)
    speech_output= Dense(hidden_dim, activation='relu')(speech_output)
    ##### Text BiLSTM-SA
    text_input = Input(shape=(len(rnn_text[0]),len(rnn_text[0][0])),dtype='float32')
    text_layer = Bidirectional(LSTM(hidden_lstm_text,return_sequences=True))(text_input)
    text_att   = Dense(hidden_con, activation='tanh')(text_layer)
    text_att_source = np.zeros((len(rnn_text),hidden_con))
    text_att_input  = Input(shape=(hidden_con,), dtype='float32')
    text_att_vec    = Dense(hidden_con,activation='relu')(text_att_input)
    text_att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([text_att_vec,text_att])
    text_att_vec = Dense(len(rnn_text[0]),activation='softmax')(text_att_vec)
    text_att_vec = layers.Reshape((len(rnn_text[0]),1))(text_att_vec)
    text_output  = layers.multiply([text_att_vec,text_layer])
    text_output  = Lambda(lambda x: K.sum(x, axis=1))(text_output)
    text_output  = Dense(hidden_dim, activation='relu')(text_output)
    ##### Total output
    output    = layers.concatenate([speech_output, text_output])
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    main_output = Dense(int(max(train_y)+1),activation='softmax')(output)
    model = Sequential()
    model = Model(inputs=[speech_input,speech_att_input,text_input,text_att_input], outputs=[main_output])
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    #####
    callbacks_list = [metricsf1macro_4input,checkpoint]
    model.summary()
    #####
    model.fit([rnn_speech,speech_att_source,rnn_text,text_att_source],train_y,validation_split = val_sp,epochs=100,batch_size= bat_size,callbacks=callbacks_list,class_weight=cw)

    validate_speech_self_text_self(total_speech,total_rec_char,total_label,64,64,32,128,class_weights,0.1,16,'model_icassp/total_multi_bilstm_att_char')
```

## 5. Multi-hop attention

```python
def validate_speech_self_text_self_mha_a(rnn_speech,rnn_text,train_y,hidden_lstm_speech,hidden_con,hidden_lstm_text,hidden_dim,cw,val_sp,bat_size,filename):
    ##### Speech BiLSTM-SA
    speech_input = Input(shape=(len(rnn_speech[0]),len(rnn_speech[0][0])), dtype='float32')
    speech_fw, speech_fw_h, speech_fw_c = LSTM(hidden_lstm_speech, return_state=True, return_sequences=True)(speech_input)
    speech_bw, speech_bw_h, speech_bw_c = LSTM(hidden_lstm_speech, return_state=True, return_sequences=True,go_backwards=True)(speech_input)
    speech_layer = layers.concatenate([speech_fw,speech_bw])
    speech_final = layers.concatenate([speech_fw_h,speech_bw_h])
    speech_att   = Dense(hidden_con, activation='tanh')(speech_layer)
    speech_att_source = np.zeros((len(rnn_speech),hidden_con))
    speech_att_input = Input(shape=(hidden_con,),dtype='float32')
    speech_att_vec = Dense(hidden_con, activation='relu')(speech_att_input)
    speech_att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([speech_att_vec,speech_att])
    speech_att_vec = Dense(len(rnn_speech[0]),activation='softmax')(speech_att_vec)
    speech_att_vec = layers.Reshape((len(rnn_speech[0]),1))(speech_att_vec)
    speech_output= layers.multiply([speech_att_vec,speech_layer])
    speech_output= Lambda(lambda x: K.sum(x, axis=1))(speech_output)
    speech_output= Dense(hidden_dim, activation='relu')(speech_output)
    ##### Text BiLSTM-SA with Speech HL output as an attention source
    text_input = Input(shape=(len(rnn_text[0]),len(rnn_text[0][0])),dtype='float32')
    text_layer = Bidirectional(LSTM(hidden_lstm_text,return_sequences=True))(text_input)
    text_att = Dense(hidden_con, activation='tanh')(text_layer)
    text_att_source = np.zeros((len(rnn_text),hidden_con))        # Dummy code
    text_att_input  = Input(shape=(hidden_con,), dtype='float32') # Dummy code
    text_att_vec    = Dense(hidden_con,activation='relu')(speech_final)	
    text_att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([text_att_vec,text_att])
    text_att_vec = Dense(len(rnn_text[0]),activation='softmax')(text_att_vec)
    text_att_vec = layers.Reshape((len(rnn_text[0]),1))(text_att_vec)
    text_output  = layers.multiply([text_att_vec,text_layer])
    text_output  = Lambda(lambda x: K.sum(x, axis=1))(text_output)
    text_output  = Dense(hidden_dim, activation='relu')(text_output)
    ##### Total output
    output    = layers.concatenate([speech_output, text_output])
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    main_output = Dense(int(max(train_y)+1),activation='softmax')(output)
    model = Sequential()
    model = Model(inputs=[speech_input,speech_att_input,text_input,text_att_input], outputs=[main_output])
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    #####
    callbacks_list = [metricsf1macro_4input,checkpoint]
    model.summary()
    #####
    model.fit([rnn_speech,speech_att_source,rnn_text,text_att_source],train_y,validation_split = val_sp,epochs=100,batch_size= bat_size,callbacks=callbacks_list,class_weight=cw)


validate_speech_self_text_self_mha_a(total_speech,total_rec_char,total_label,64,64,32,128,class_weights,0.1,16,'model_icassp_temp/total_mha_a_att_char')
```

```python
def validate_speech_self_text_self_mha_a_t(rnn_speech,rnn_text,train_y,hidden_lstm_speech,hidden_con,hidden_lstm_text,hidden_dim,cw,val_sp,bat_size,filename):
    ##### Speech BiLSTM
    speech_input = Input(shape=(len(rnn_speech[0]),len(rnn_speech[0][0])), dtype='float32')
    speech_fw, speech_fw_h, speech_fw_c = LSTM(hidden_lstm_speech, return_state=True, return_sequences=True)(speech_input)
    speech_bw, speech_bw_h, speech_bw_c = LSTM(hidden_lstm_speech, return_state=True, return_sequences=True,go_backwards=True)(speech_input)
    speech_layer = layers.concatenate([speech_fw,speech_bw])
    speech_final = layers.concatenate([speech_fw_h,speech_bw_h])
    ##### Text BiLSTM-SA with Speech HL output as an attention source
    text_input = Input(shape=(len(rnn_text[0]),len(rnn_text[0][0])),dtype='float32')
    text_layer = Bidirectional(LSTM(hidden_lstm_text,return_sequences=True))(text_input)
    text_att = Dense(hidden_con, activation='tanh')(text_layer)
    text_att_source = np.zeros((len(rnn_text),hidden_con))        # Dummy code
    text_att_input  = Input(shape=(hidden_con,), dtype='float32') # Dummy code
    text_att_vec    = Dense(hidden_con,activation='relu')(speech_final)	
    text_att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([text_att_vec,text_att])
    text_att_vec = Dense(len(rnn_text[0]),activation='softmax')(text_att_vec)
    text_att_vec = layers.Reshape((len(rnn_text[0]),1))(text_att_vec)
    text_output  = layers.multiply([text_att_vec,text_layer])
    text_output  = Lambda(lambda x: K.sum(x, axis=1))(text_output)
    text_output  = Dense(hidden_dim, activation='relu')(text_output)
    ##### Speech BiLSTM-SA with Speech-Text HL output as an attention source
    speech_att   = Dense(hidden_con, activation='tanh')(speech_layer)
    speech_att_source = np.zeros((len(rnn_speech),hidden_con))    # Dummy code
    speech_att_input = Input(shape=(hidden_con,),dtype='float32') # Dummy code
    speech_att_vec = Dense(hidden_con, activation='relu')(text_output)
    speech_att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([speech_att_vec,speech_att])
    speech_att_vec = Dense(len(rnn_speech[0]),activation='softmax')(speech_att_vec)
    speech_att_vec = layers.Reshape((len(rnn_speech[0]),1))(speech_att_vec)
    speech_output= layers.multiply([speech_att_vec,speech_layer])
    speech_output= Lambda(lambda x: K.sum(x, axis=1))(speech_output)
    speech_output= Dense(hidden_dim, activation='relu')(speech_output)
    ##### Total output
    output    = layers.concatenate([speech_output, text_output])
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    main_output = Dense(int(max(train_y)+1),activation='softmax')(output)
    model = Sequential()
    model = Model(inputs=[speech_input,speech_att_input,text_input,text_att_input], outputs=[main_output])
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    #####
    callbacks_list = [metricsf1macro_4input,checkpoint]
    model.summary()
    #####
    model.fit([rnn_speech,speech_att_source,rnn_text,text_att_source],train_y,validation_split = val_sp,epochs=100,batch_size= bat_size,callbacks=callbacks_list,class_weight=cw)


validate_speech_self_text_self_mha_a_t(total_speech,total_rec_char,total_label,64,64,32,128,class_weights,0.1,16,'model_icassp_temp/total_mha_a_t_drop_att_char')
```

## 6. Cross-attention

```python
def validate_speech_self_text_self_ca_mod(rnn_speech,rnn_text,train_y,hidden_lstm_speech,hidden_con,hidden_lstm_text,hidden_dim,cw,val_sp,bat_size,filename):
    ##### Speech BiLSTM
    speech_input = Input(shape=(len(rnn_speech[0]),len(rnn_speech[0][0])), dtype='float32')
    speech_layer = Bidirectional(LSTM(hidden_lstm_speech,return_sequences=True))(speech_input)
    speech_att   = Dense(hidden_con, activation='tanh')(speech_layer)
    speech_att_source= np.zeros((len(rnn_speech),hidden_con))
    speech_att_input = Input(shape=(hidden_con,),dtype='float32')
    speech_att_vec   = Dense(hidden_con, activation='relu')(speech_att_input)
    speech_att_vec   = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([speech_att_vec,speech_att])
    speech_att_vec   = Dense(len(rnn_speech[0]),activation='softmax')(speech_att_vec)
    speech_att_vec   = layers.Reshape((len(rnn_speech[0]),1))(speech_att_vec)
    speech_output= layers.multiply([speech_att_vec,speech_layer])
    speech_output= Lambda(lambda x: K.sum(x, axis=1))(speech_output)
    speech_output= Dense(hidden_dim, activation='relu')(speech_output)
    ##### Text BiLSTM
    text_input = Input(shape=(len(rnn_text[0]),len(rnn_text[0][0])),dtype='float32')
    text_fw, text_fw_h, text_fw_c = LSTM(hidden_lstm_text, return_state=True, return_sequences=True)(text_input)
    text_bw, text_bw_h, text_bw_c = LSTM(hidden_lstm_text, return_state=True, return_sequences=True,go_backwards=True)(text_input)
    text_layer = layers.concatenate([text_fw,text_bw])
    text_final = layers.concatenate([text_fw_h,text_bw_h])
    text_att   = Dense(hidden_con, activation='tanh')(text_layer)	
    text_att_source = np.zeros((len(rnn_text),hidden_con))        # Dummy code
    text_att_input  = Input(shape=(hidden_con,), dtype='float32') # Dummy code	
    ##### Exchange phase
    speech_att_hop = Dense(hidden_con, activation='relu')(text_final)	
    speech_att_hop = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([speech_att_hop,speech_att])
    speech_att_hop = Dense(len(rnn_speech[0]),activation='softmax')(speech_att_hop)
    speech_att_hop = layers.Reshape((len(rnn_speech[0]),1))(speech_att_hop)	
    speech_output_hop = layers.multiply([speech_att_hop,speech_layer])
    speech_output_hop = Lambda(lambda x: K.sum(x, axis=1))(speech_output_hop)
    speech_output_hop = Dense(hidden_dim, activation='relu')(speech_output_hop)
    text_att_hop = Dense(hidden_con, activation='relu')(speech_output)	
    text_att_hop = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([text_att_hop,text_att])
    text_att_hop = Dense(len(rnn_text[0]),activation='softmax')(text_att_hop)
    text_att_hop = layers.Reshape((len(rnn_text[0]),1))(text_att_hop)	
    text_output_hop = layers.multiply([text_att_hop,text_layer])
    text_output_hop = Lambda(lambda x: K.sum(x, axis=1))(text_output_hop)
    text_output_hop = Dense(hidden_dim, activation='relu')(text_output_hop)	
    ##### Total output
    output    = layers.concatenate([speech_output_hop, text_output_hop])
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    main_output = Dense(int(max(train_y)+1),activation='softmax')(output)
    model = Sequential()
    model = Model(inputs=[speech_input,speech_att_input,text_input,text_att_input], outputs=[main_output])
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    #####
    callbacks_list = [metricsf1macro_4input,checkpoint]
    model.summary()
    #####
    model.fit([rnn_speech,speech_att_source,rnn_text,text_att_source],train_y,validation_split = val_sp,epochs=100,batch_size= bat_size,callbacks=callbacks_list,class_weight=cw)


validate_speech_self_text_self_ca_mod(total_speech,total_rec_char,total_label,64,64,32,128,class_weights,0.1,16,'model_icassp_temp/total_ca_mod_att_char')
```
