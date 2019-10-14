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

Understanding the intention of an utterance is challenging for some prosody-sensitive cases, especially when it is in the written form as in a text chatting or speech recognition output. The main concern is detecting the directivity or rhetoricalness of an utterance and distinguishing the type of question. Since it is inevitable to face both the issues regarding prosody and semantics, the identification is expected be benefited from the observations on human language processing mechanism. 

Challenging issues for spoken language understanding (SLU) modules include inferring the intention of syntactically ambiguous utterances. If an utterance has an underspecified sentence ender whose role is decided only upon prosody, the inference requires whole the acoustic and textual data of the speech for SLUs (and even human) to correctly infer the intention, since the pitch sequence, the duration between the words, and the overall tone decides the intention of the utterance. For example, in Seoul Korean which is *wh-in-situ*, many sentences incorporate various ways of interpretation that depend on the intonation, as shown in our [ICPhS paper](http://www.assta.org/proceedings/ICPhS2019/papers/ICPhS_3951.pdf).

Here, we attack the issue above utilizing the speech corpus that is distributed along with the paper. The scripts and speech files are available in [this github repository](https://github.com/warnikchow/prosem). As you download the folder from [the dropbox](https://www.dropbox.com/s/3tm6ylu21jpmnj8/ProSem_KOR_speech.zip?dl=0), unzip the folder in YOUR DIRECTORY so that you have *ProSem_KOR_speech* folder there. In it, there are the folders named *FEMALE* and *MALE* each containing 3,551 Korean speech utterances. Also, place the folder *text* in YOUR DIRECTORY, which contains the scripts of the speech files.

## 1. Extracting acoustic features

## 2. Speech-only analysis with Librosa and Keras

## 3. Self-attentive BiLSTM

## 4. Parallel utilization of audio and text data

## 5. Multi-hop attention

## 6. Cross-attention
