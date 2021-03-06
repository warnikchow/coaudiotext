# CoAudioText
**A short tutorial for the co-utilization of audio and text data (multi-modal analysis)**

## Contents
[0. Problem definition & loading dataset](https://github.com/warnikchow/coaudiotext/blob/master/README.md#0-problem-definition--loading-dataset)</br>
[1. Extracting acoustic features](https://github.com/warnikchow/coaudiotext/blob/master/README.md#1-extracting-acoustic-features)</br>
[2. Speech-only analysis with Librosa and Keras](https://github.com/warnikchow/coaudiotext/blob/master/README.md#2-speech-only-analysis-with-librosa-and-keras)</br>
[3. Self-attentive BiLSTM](https://github.com/warnikchow/coaudiotext/blob/master/README.md#3-self-attentive-bilstm)</br>
[4. Parallel utilization of audio and text data](https://github.com/warnikchow/coaudiotext/blob/master/README.md#4-parallel-utilization-of-audio-and-text-data)</br>
[5. Multi-hop attention](https://github.com/warnikchow/coaudiotext/blob/master/README.md#5-multi-hop-attention)</br>
[6. Cross-attention](https://github.com/warnikchow/coaudiotext/blob/master/README.md#6-cross-attention)</br>
[7. Result and analysis?](https://github.com/warnikchow/coaudiotext#7-result-and-analysis)

## 0. Problem definition & loading dataset
#### The introduction partly comes from [our CogSci paper](https://arxiv.org/abs/1910.09275)!

Understanding the intention of an utterance is challenging for some prosody-sensitive cases, especially when it is in the written form, as in a text chatting or speech recognition output. **The main concern is detecting the directivity or rhetoricalness of an utterance and distinguishing the type of question.** Since it is inevitable to face both the issues regarding prosody and semantics, the identification is expected to be benefited from the observations on the human language processing mechanism. 

In this project, we handle one of less explored issues in spoken language understanding (SLU): **inferring the intention of syntactically ambiguous utterances**. If an utterance has an underspecified sentence ender whose role is decided only upon prosody, the inference requires whole the acoustic and textual data of the speech for SLUs (and even human) to correctly infer the intention, since the pitch sequence, the duration between the words, and the overall tone decide the intention of the utterance. For example, in Seoul Korean, which is *wh-in-situ*, many sentences incorporate various ways of interpretation that depend on the intonation, as shown in our [ICPhS paper](http://www.assta.org/proceedings/ICPhS2019/papers/ICPhS_3951.pdf).

<p align="center">
    <image src="https://github.com/warnikchow/coaudiotext/blob/master/images/fig.png" width="500"></br>
          <strong>Prosody-semantics interface in Seoul Korean</strong>

Here, we attack the issue above, utilizing the speech corpus that is distributed along with the paper. First, *git clone* this repository, *pip install -r Requirements.txt* and let it be YOUR DIRECTORY. It then contains the folder *text*, which contains the scripts of the speech files, and *han2one.py* that contains the function that converts the Korean characters to multi-hot vectors. The speech files are available in [this github repository](https://github.com/warnikchow/prosem). As you download the folder from [the dropbox](https://www.dropbox.com/s/3tm6ylu21jpmnj8/ProSem_KOR_speech.zip?dl=0), unzip the folder in YOUR DIRECTORY so that you have *ProSem_KOR_speech* folder there. In it, there are the folders named *FEMALE* and *MALE* each containing 3,551 Korean speech utterances. If you add another folder *model* in YOUR DIRECTORY to save your trained networks, every setting is over. In summary, **YOUR DIRECTORY may contain *han2one.py*, *text*, *ProSem_KOR_speech*, and *model***.

*This tutorial is processed line-by-line, thus start with **python** (best if 3.5.2) in bash!* 

## 1. Extracting acoustic features

**First, since we only utilize the utterances that can be disambiguated with speech, here we extract the acoustic features from the files. There are many ways to abstract the physical components, but here we utilize [mel spectrogram](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0) due to its rich acoustic information and intuitive concept. The process is done with [Librosa](https://librosa.github.io/librosa/), a python-based audio signal processing library.**

```python
import numpy as np
import librosa

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

##### Reads the data from a text file that contains intention labels and scripts
##### Seven labels are utilized in total, and each utterance corresponds to 2 ~ 4 labels depending on the prosody
##### Scripts are all the same for the ambiguous utterances
text_total = read_data('text/spec_final.txt')
labels = ['s','yn','wh','rq','c','r','rc']
text_answer_pair = [[t[2],labels.index(t[3])] for t in text_total]

##### The array that contains the shuffled indices of the utterances
x_fem = np.load('text/x_fem.npy')
x_mal = np.load('text/x_mal.npy')

fem_data  = [text_answer_pair[int(n)][0] for n in x_fem]
fem_label = [text_answer_pair[int(n)][1] for n in x_fem]
mal_data  = [text_answer_pair[int(n)][0] for n in x_mal]
mal_label = [text_answer_pair[int(n)][1] for n in x_mal]

total_data_train = fem_data[:3196]+mal_data[:3196]
total_data_test  = fem_data[3196:]+mal_data[3196:]
total_data = total_data_train+total_data_test    # The data regarding text scripts, assumed perfectly transcribed

total_label_train = fem_label[:3196]+mal_label[:3196]
total_label_test  = fem_label[3196:]+mal_label[3196:]
total_label = total_label_train+total_label_test # The data regarding the gold intention labels, 0 to 6

##### Acoustic featurization
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
        rmse = np.transpose(rmse)     # Normalized RMSE sequence augmented to reflect syllable-timedness
        S = librosa.feature.melspectrogram(S=D)
        S = np.transpose(S)           # Conventional mel spectrogram
        if len(S)>=mlen:              # Max frame length about 200 (~7s)
            data_s_rmse[i][:,0]=rmse[-mlen:,0]
            data_s_rmse[i][:,1:]=S[-mlen:,:]
        else:                         # Padding from the back part, considering head-finality
            data_s_rmse[i][-len(S):,0]=np.transpose(rmse)
            data_s_rmse[i][-len(S):,1:]=S
    return data_s_rmse

fem_speech = make_data('ProSem_KOR_speech/FEMALE/',3552,x_fem,200)
mal_speech = make_data('ProSem_KOR_speech/MALE/',3552,x_mal,200)

total_speech_train = np.concatenate([fem_speech[:3196],mal_speech[:3196]])
total_speech_test  = np.concatenate([fem_speech[3196:],mal_speech[3196:]])
total_speech = np.concatenate([total_speech_train,total_speech_test]) # The acoustic features of the speech files
```

**Note that here, for every speech file, a feature of length 200 is yielded, with the width 129. 128 corresponds to the mel spectrogram, and the left one denotes an RMSE of each frame. The latter was attached to effectively represent the syllable-level discreteness of Korean spoken language, as suggested in [the previous experiment regarding intonation type identification](https://github.com/warnikchow/korinto).**

## 2. Speech-only analysis with Librosa and Keras

**Although people these days seem to migrate to TF2.0 and PyTorch, I still use [original Keras](https://keras.io/) for its conciseness. Hope this code fit to whatever flatform the reader uses.**

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

**The next step is required for a balanced training and evaluation. *class_weights* denote the ratio that is to be weighted in training phase regarding the corpus utterance type volume, and by defining *F1 score*, we can get the point of the evaluation which *accuracy* usually fails to discern while using the imbalanced corpus.**

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

<p align="center">
    <image src="https://github.com/warnikchow/coaudiotext/blob/master/images/bilstm.png" width="600"></br>
          <strong>Concept of BiLSTM from http://www.gabormelli.com/RKB/Bidirectional_LSTM_(biLSTM)_Model</strong>

**The following denotes how we define the BiLSTM by using Keras, although no functional API is utilized here. We use only *Sequential()*, and no more complex structure is used. We don't use dropout here, since considering the hidden layers of this size, overhead is less expected.**

```python
##### Vanilla BiLSTM utilizes only Sequential() module of Keras
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

validate_bilstm(total_speech,total_label,64,128,class_weights,0.1,16,'model/total_bilstm')
```

## 3. Self-attentive BiLSTM

**Remember: mel spectrogram still has plenty of prosody-semantic information! Thus, we decided to apply a self-attentive embedding which has been successfully used in text processng. Before making up the module, in view of *pure Keras* where F1 measure is removed (well if the recent version has one, that's nice!), we need another definition of F1 score since additional input source is introduced (zero vector for attention source initialization).**

```python
##### f1-2input
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

<p align="center">
    <image src="https://github.com/warnikchow/coaudiotext/blob/master/images/sa.png" width="600"></br>
          <strong>Self-attentive embedding for sentence representation (Lin et al., 2017)</strong>

**And here we define our self-attentive BiLSTM model which *sometimes* uses TensorFlow backend. This kind of design (utilizing *Model* module) is inevitable since the *pure* Keras approach cannot guarantee that we can make up such a complicated layer... So, rather detailed comments are attached to help the readers follow how the structure (above) in [the paper](https://arxiv.org/abs/1703.03130) is implemented as a code.**

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

validate_rnn_self_drop(total_speech,total_label,64,64,128,class_weights,0.1,16,'model/total_bilstm_att')
```

## 4. Parallel utilization of audio and text data

**The next step is to finally adopt the textual features that can bring the lexical meanings into the speech analysis. So far, we've used the BiLSTM network that only exploits audio features, but here we make a representation for the sentences so that we can embed the input text and co-utilize it in the inference. The concatenation is similar to the network (below) suggested in [the paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6261374/) that dealt with multimodal speech intention understanding, but the detailed architecture differs.**

<p align="center">
    <image src="https://github.com/warnikchow/coaudiotext/blob/master/images/para.jpg" width="600"></br>
          <strong>The concatenated architecture we referred for Para-BRE-Att (Gu et al., 2017)</strong>

**The character-level text embedding is quite different from English, but instead of either feature-based or fine-tuning approaches, here we utilize the [multi-hot encoding](https://www.researchgate.net/publication/331987503_Sequence-to-Sequence_Autoencoder_based_Korean_Text_Error_Correction_using_Syllable-level_Multi-hot_Vector_Representation) that was [shown to be useful in Korean sentence classification](https://arxiv.org/abs/1905.13656). All the characters are represented into a 67-dim sparse vector with 2-3 non-zero terms, and the full text feature has size 30 x 67. The maximum length 30 is enough for the experiment considering the property of the dataset. Refer to [this repository](https://github.com/warnikchow/kcharemb) for other types of Korean character-level embedding! Well, at least at this point, we're going to use the type of character-level encoding that is as concise as possible, not heavy, and notwithstanding informative.**

***Note that this process is very, very language-dependent. You can replace this procedure with whatever corpus and tokenization you utilize. Just make sure that the format fits as a BiLSTM input.***

```python
import hgtk
import han2one
from han2one import shin_onehot, cho_onehot, char2onehot
alp = han2one.alp
uniquealp = han2one.uniquealp

##### Featurization for character-level multi-hot (sparse) representation
def featurize_rnn_only_char(corpus,maxlen):
    rnn_char  = np.zeros((len(corpus),maxlen,len(alp)))
    for i in range(len(corpus)):
        if i%1000 ==0:
            print(i)
        s = corpus[i]
        for j in range(len(s)):
            if j < maxlen and hgtk.checker.is_hangul(s[-j-1])==True:
                rnn_char[i][-j-1,:] = char2onehot(s[-j-1])   # The characters are padded from back end, as in audio
    return rnn_char
    
total_rec_char = featurize_rnn_only_char(total_data,30)      # Maximum character length = 30 (sufficient)
```

**Next, we should take into account that the number of inputs gets bigger again; this time to four - that we should define another class for evaluating the F1 score. It would have been best for us to put together these kinds of materials in a single *.py* file and import it. Well, the specification will be modified as this tutorial gets more organized.**

```python
##### f1-4input

class Metricsf1macro_4input(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_f1s_w = []
        self.val_recalls_w = []
        self.val_precisions_w = []
    def on_epoch_end(self, epoch, logs={}):
        if len(self.validation_data)>2:
            val_predict = np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1],self.validation_data[2],self.validation_data[3]]))
            val_predict = np.argmax(val_predict,axis=1)
            val_targ = self.validation_data[4]
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

metricsf1macro_4input = Metricsf1macro_4input()

```

**And here comes the model architecture for our Para-BRE-Att, which incorporates two BiLSTM networks that each contains the information regarding audio and text of the speech, and then concatenation.**

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
    ##### Total output after concatenation and MLPs
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
    
validate_speech_self_text_self(total_speech,total_rec_char,total_label,64,64,32,128,class_weights,0.1,16,'model/total_multi_bilstm_att_char')    
```

**The idea is straightforward and is widely used within many algorithms nowadays, yielding an adequate performance. Also, replacing the CNN - that was held for the spectrogram in the original reference - with BiLSTM, seems to be successful for AT LEAST IN OUR DATASET. The reason is assumed to be the property regarding syntax-semantics of the task, rather than only of semantics such as in sentiment analysis.**

## 5. Multi-hop attention

**In this section, multi-hop attention that was [previously proposed for emotion recognition](https://arxiv.org/abs/1904.10788), is implemented in Keras and is applied to our task, speech intention disambiguation. The first module incorporates only one hopping, from audio representation output to the text features' hidden layers. The picture below may help you understand how hopping occurs, in a very intuitive way. Slightly different from the original paper, we've named MHA-1 as MHA-A, and MHA-2 as MHA-AT, to reflect the features that are utilized.**

<p align="center">
    <image src="https://github.com/warnikchow/coaudiotext/blob/master/images/mha.PNG" width="700"></br>
          <strong>A simple BRE (BiLSTM), and three parallel variations namely MHA-1,2,3 (Yoon et al., 2019)</strong>

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
    speech_output= Lambda(lambda x: K.sum(x, axis=1))(speech_output)    # Concatenated audio feature is already assigned weight
    speech_output= Dense(hidden_dim, activation='relu')(speech_output)
    ##### Text BiLSTM-SA with Speech HL output as an attention source
    text_input = Input(shape=(len(rnn_text[0]),len(rnn_text[0][0])),dtype='float32')
    text_layer = Bidirectional(LSTM(hidden_lstm_text,return_sequences=True))(text_input)
    text_att = Dense(hidden_con, activation='tanh')(text_layer)
    text_att_source = np.zeros((len(rnn_text),hidden_con))              # Dummy code
    text_att_input  = Input(shape=(hidden_con,), dtype='float32')       # Dummy code
    text_att_vec    = Dense(hidden_con,activation='relu')(speech_final)	# Hopping happens here!
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


validate_speech_self_text_self_mha_a(total_speech,total_rec_char,total_label,64,64,32,128,class_weights,0.1,16,'model/total_mha_a_att_char')
```

**Note that the dummy code was commented to denote that the line was not removed to guarantee the same input formats. The next chunk involves another hopping.** 

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
    text_att_source = np.zeros((len(rnn_text),hidden_con))              # Dummy code
    text_att_input  = Input(shape=(hidden_con,), dtype='float32')       # Dummy code
    text_att_vec    = Dense(hidden_con,activation='relu')(speech_final)	# First hopping Audio >> Text
    text_att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([text_att_vec,text_att])
    text_att_vec = Dense(len(rnn_text[0]),activation='softmax')(text_att_vec)
    text_att_vec = layers.Reshape((len(rnn_text[0]),1))(text_att_vec)
    text_output  = layers.multiply([text_att_vec,text_layer])
    text_output  = Lambda(lambda x: K.sum(x, axis=1))(text_output)
    text_output  = Dense(hidden_dim, activation='relu')(text_output)
    ##### Speech BiLSTM-SA with Speech-Text HL output as an attention source
    speech_att   = Dense(hidden_con, activation='tanh')(speech_layer)
    speech_att_source = np.zeros((len(rnn_speech),hidden_con))          # Dummy code
    speech_att_input = Input(shape=(hidden_con,),dtype='float32')       # Dummy code
    speech_att_vec = Dense(hidden_con, activation='relu')(text_output)  # Second hopping (Audio >>) Text >> Audio
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


validate_speech_self_text_self_mha_a_t(total_speech,total_rec_char,total_label,64,64,32,128,class_weights,0.1,16,'model/total_mha_a_t_drop_att_char')
```

## 6. Cross-attention

**The last step is building up a cross-attention network, which was inspired by [the implementation regarding image-text matching](https://kuanghuei.github.io/SCANProject/). Due to the different nature of speech and image, we've utilized a slightly different type of architecture; but the philosophy still holds. Rather than giving attention to a single feature at a time and doing that to the other subsequentially, how about observing both of them simultaneously?**

<p align="center">
    <image src="https://github.com/warnikchow/coaudiotext/blob/master/images/ca.PNG" width="700"></br>
          <strong>The concept of cross-attention, though the illustration is for vision domain (Lee et al., 2018)</strong>
    
```python
def validate_speech_self_text_self_ca(rnn_speech,rnn_text,train_y,hidden_lstm_speech,hidden_con,hidden_lstm_text,hidden_dim,cw,val_sp,bat_size,filename):
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
    ##### The attention source is simply the final hidden layer, not the weight summed sequence
    ##### This kind of implementation was done empirically, upon the performance
    text_input = Input(shape=(len(rnn_text[0]),len(rnn_text[0][0])),dtype='float32')
    text_fw, text_fw_h, text_fw_c = LSTM(hidden_lstm_text, return_state=True, return_sequences=True)(text_input)
    text_bw, text_bw_h, text_bw_c = LSTM(hidden_lstm_text, return_state=True, return_sequences=True,go_backwards=True)(text_input)
    text_layer = layers.concatenate([text_fw,text_bw])
    text_final = layers.concatenate([text_fw_h,text_bw_h])
    text_att   = Dense(hidden_con, activation='tanh')(text_layer)	
    text_att_source = np.zeros((len(rnn_text),hidden_con))              # Dummy code
    text_att_input  = Input(shape=(hidden_con,), dtype='float32')       # Dummy code	
    ##### Exchange phase
    speech_att_hop = Dense(hidden_con, activation='relu')(text_final)	
    speech_att_hop = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([speech_att_hop,speech_att])
    speech_att_hop = Dense(len(rnn_speech[0]),activation='softmax')(speech_att_hop)
    speech_att_hop = layers.Reshape((len(rnn_speech[0]),1))(speech_att_hop)	
    speech_output_hop = layers.multiply([speech_att_hop,speech_layer])  # Text-influenced attention for audio
    speech_output_hop = Lambda(lambda x: K.sum(x, axis=1))(speech_output_hop)
    speech_output_hop = Dense(hidden_dim, activation='relu')(speech_output_hop)
    text_att_hop = Dense(hidden_con, activation='relu')(speech_output)	
    text_att_hop = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([text_att_hop,text_att])
    text_att_hop = Dense(len(rnn_text[0]),activation='softmax')(text_att_hop)
    text_att_hop = layers.Reshape((len(rnn_text[0]),1))(text_att_hop)	
    text_output_hop = layers.multiply([text_att_hop,text_layer])        # Audio-influenced attention for text
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


validate_speech_self_text_self_ca(total_speech,total_rec_char,total_label,64,64,32,128,class_weights,0.1,16,'model/total_ca_mod_att_char')
```

**At this point, we think that summarizing the specification and hyperparameters of our implementation will be helpful to the readers.</br> - Device: NVidea Titan V</br> - Batch size: 16</br> - Width for the hidden layers of BiLSTM:  64 * 2 = 128</br> - Width for MLPs: 128</br> - Width of the context vector: 64</br> - Dropout rate: 0.3</br> - Activation: *tanh* for the context vectors, *softmax* for the decision, and *ReLU* for others</br> - Optimizer: Adam (0.0005) </br>- BiLSTM and MLPs implemented by Keras as TF backend**

**The below is the wrap-up for all the models implemented so far. God thanks *.pptx*!**

<p align="center">
    <image src="https://github.com/warnikchow/coaudiotext/blob/master/images/diagram.png" width="700"></br>
          <strong>The block diagrams of the implemented models</br>
    (1) Speech-only: Audio-BRE </br>
    (2) Self-attentive for Speech-only: Audio-BRE-Att </br>
    (3) Text-aided: Para-BRE-Att </br>
    (4) Multi-hop Attention: MHA-A,AT</br>
    (5) Cross-attention: CA </strong> 

## 7. Result and analysis?

<p align="center">
    <image src="https://github.com/warnikchow/coaudiotext/blob/master/images/table.PNG" width="500"></br>
         
**Result on the 10% test set. For each feature, the intersection was chosen among 5-best accuracy and F1 models that were yielded during first 100 epochs of training. Details are currently available in [the paper](https://arxiv.org/abs/1910.09275), and to be supplemented here afterward. But for TL;DR:** 

### ***It is assumed that speech intention analysis is affected dominantly by the combination of speech analysis and speech-aided text analysis, preferably with the smaller contribution of text-aided speech analysis.***

### MAINTAINER
[Won Ik Cho](https://sites.google.com/site/warnikchow) (aka WarNik Chow)</br>
[@Human Interface Laboratory](https://hi.snu.ac.kr/), Dept. of ECE and INMC, Seoul National University, Seoul, Korea</br>
If there is any issue, feel free to contact by wicho@hi.snu.ac.kr. 

### ACKNOWLEDGEMENT
The great thanks go to [Jeonghwa Cho](https://sites.google.com/site/jeonghwacho01), who's discussed the topic (syntactic ambiguity) together from ever since this project started, given helpful ideas on psycholinguistics, and co-organized [the very first ICPhS paper](http://www.assta.org/proceedings/ICPhS2019/papers/ICPhS_3951.pdf). The author also thanks Jeemin Kang and Woo Hyun Kang, who respectively checked [the original dataset](https://github.com/warnikchow/prosem) and proofread [this paper](https://arxiv.org/abs/1910.09275). The reference and further correspondence goes to Nam Soo Kim, my advisor, and this project was largely supported by Technology Innovation Program (10076583, Development of free-running speech recognition technologies for embedded robot system) funded by the Ministry of Trade, Industry & Energy (MOTIE, Korea).

### CITATION
If you want to utilize the analysis result or code of this project, the following is a placeholder for the preprint:
```
@article{cho2019text,
    title={Text Matters but Speech Influences: A Computational Analysis of Syntactic Ambiguity Resolution},
    author={Cho, Won Ik and Cho, Jeonghwa and Kang, Woo Hyun and Kim, Nam Soo},
    journal={arXiv preprint arXiv:1910.09275},
    year={2019}
}
```
And if you want to exploit the dataset merely, the following will be helpful:
```
@inproceedings{cho2019prosody,
	title={Prosody-semantics interface in Seoul Korean: Corpus for a disambiguation of wh-intervention},
	author={Cho, Won Ik and Cho, Jeonghwa and Kang, Jeemin and Kim, Nam Soo},
	booktitle={Proceedings of the 19th International Congress of the Phonetic Sciences (ICPhS 2019)},
	pages={3902--3906},
	year={2019}
}
```
