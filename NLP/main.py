import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense
from keras.layers import LSTM,Input
from keras.models import Model
import numpy as np
import opennmt
from opennmt import models


def GetUniqueChars(listOfSentences):
    uniqueChars = []

    for sentence in listOfSentences:
        for char in sentence:
            if char not in uniqueChars:
                uniqueChars.append(char)
                uniqueChars.sort()

    return uniqueChars


def GetOneHotEncodedSentences(sentencesList, charDictionary, charsSize, maxSentencesSize):

    oneHotEncodedData = []
    for sentenceIndex in range(debugingSize):

        charOneHotEncoding = []
        sentenceEncodingByCharacter = []

        k = len(sentencesList[sentenceIndex])
        m = 0

        # One hot encode each character at a time
        while m < k:
            for char in sentencesList[sentenceIndex][m]:

                for i in range(charsSize):

                    if charDictionary[char] == i:
                        charOneHotEncoding.append(1)
                    else:
                        charOneHotEncoding.append(0)

            sentenceEncodingByCharacter.append(charOneHotEncoding)
            charOneHotEncoding = []
            m = m + 1

        # Fill the rest(until max size of sentence) with 1 in the beggining
        while m < maxSentencesSize:
            for i in range(charsSize):
                if i == 0:
                    charOneHotEncoding.append(1)
                else:
                    charOneHotEncoding.append(0)

            sentenceEncodingByCharacter.append(charOneHotEncoding)
            charOneHotEncoding = []
            m = m + 1

        oneHotEncodedData.append(sentenceEncodingByCharacter)

    return oneHotEncodedData

# def GetOneHotEncodeEnglishExpressionForTranslation():
#     print("fml")

projectRoot = os.getcwd()
dataSetPath = os.path.join(projectRoot,'PlaceDatasetHere','por.txt')

englishSentences = []
brazilSentences = []

debugingSize = 30000

with open(dataSetPath, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

for line in lines[:debugingSize]:
    englishSentence, brazilSentence, _ = line.split("\t")
    englishSentences.append(englishSentence)
    brazilSentences.append("\t" + brazilSentence + "\t")


englishChars = GetUniqueChars(englishSentences)
brazilChars = GetUniqueChars(brazilSentences)

englishDict={}
for i in range(len(englishChars)):
    englishDict[englishChars[i]]=i

brazilDict={}
for i in range(len(brazilChars)):
    brazilDict[brazilChars[i]]=i



english_encoder_tokens = len(englishChars)
brazil_decoder_tokens = len(brazilChars)

numberEnglishToken = len(englishChars)
numberBrazilToken = len(brazilChars)


#Gets max length of a sentence in each language for encoding and decoding
max_encoder_seq_length = 0
for sentence in englishSentences:
    if len(sentence) > max_encoder_seq_length:
        max_encoder_seq_length = len(sentence)

max_decoder_seq_length = 0
for sentence in brazilSentences:
    if len(sentence) > max_decoder_seq_length:
        max_decoder_seq_length = len(sentence)

#WORKING DO NOT MOVE
encoder_input_data = np.array(GetOneHotEncodedSentences(englishSentences,englishDict , len(englishChars),max_encoder_seq_length))
decoder_input_data = np.array(GetOneHotEncodedSentences(brazilSentences,brazilDict, len(brazilChars),max_decoder_seq_length))



print(encoder_input_data.shape)
print(decoder_input_data.shape)

decoder_target_data = []
for sentenceIndex in range(30000):
    a = []
    b = []
    c = []
    k = len(brazilSentences[sentenceIndex])
    m = 1
    while m < k:
        for char in brazilSentences[sentenceIndex][m]:
            for i in range(len(brazilChars)):
                if brazilDict[char] == i:
                    a.append(1)
                else:
                    a.append(0)

        for kp in a:
            b.append(kp)
        c.append(b)
        b = []
        a = []
        m = m + 1
    m = m - 1
    while m < max_decoder_seq_length:
        for i in range(len(brazilChars)):
            if i == 0:
                a.append(1)
            else:
                a.append(0)


        for kp in a:
            b.append(kp)
        c.append(b)
        b = []
        a = []
        m = m + 1
    decoder_target_data.append(c)
decoder_target_data = np.array(decoder_target_data)



#https://www.kaggle.com/code/jayantawasthi/languagetranslater-english-to-french - Model taken from this project

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256

encoder_inputs = Input(shape=(None, len(englishChars)))
encoder = LSTM(latent_dim, dropout=0.2, return_sequences=True, return_state=True)
encoder_outputs_1, state_h_1, state_c_1 = encoder(encoder_inputs)
encoder = LSTM(latent_dim, dropout=0.2, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_outputs_1)
encoder_states = [state_h_1, state_c_1, state_h, state_c]


decoder_inputs = Input(shape=(None, len(brazilChars)))
decoder_lstm = LSTM(latent_dim, return_sequences=True, dropout=0.2, return_state=True)
decoder_outputs_1, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h_1, state_c_1])
decoder_lstm_1 = LSTM(latent_dim, return_sequences=True, dropout=0.2, return_state=True)
decoder_outputs, _, _ = decoder_lstm_1(decoder_outputs_1, initial_state=[state_h, state_c])
decoder_dense = Dense(len(brazilChars), activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)

model.save("engtobr.h5")

#LOAD MODEL INSTEAD OF TRAINING
# modelName = "engtobr.h5"
# modelPath = os.path.join(projectRoot,modelName)
# model = keras.models.load_model(modelPath)
# encoder_inputs = model.input[0]  # input_1
#
# encoder_outputs_1, state_h_enc_1, state_c_enc_1 = model.layers[2].output
# encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output
# encoder_states = [state_h_enc_1, state_c_enc_1, state_h_enc, state_c_enc]
# encoder_model_1 = keras.Model(encoder_inputs, encoder_states)
#
# decoder_inputs = model.input[1]
# decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
# decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
# decoder_state_input_h1 = Input(shape=(latent_dim,), name="input_5")
# decoder_state_input_c1 = Input(shape=(latent_dim,), name="input_6")
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c, decoder_state_input_h1,
#                          decoder_state_input_c1]
# decoder_lstm = model.layers[3]
# dec_o, state_h, state_c = decoder_lstm(
#     decoder_inputs, initial_state=decoder_states_inputs[:2])
# decoder_lstm_1 = model.layers[5]
# dec_o_1, state_h1, state_c1 = decoder_lstm_1(
#     dec_o, initial_state=decoder_states_inputs[-2:])
# decoder_states = [state_h, state_c, state_h1, state_c1]
# decoder_dense = model.layers[6]
# decoder_outputs = decoder_dense(dec_o_1)
# decoder_model = keras.Model(
#     [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
# )
#--------------------------------------------------------------------------------------------------------


reverse_input_char_index = {}
for i in range(len(englishChars)):
    reverse_input_char_index[i] = englishChars[i]

reverse_target_char_index = {}
for i in range(len(brazilChars)):
    reverse_target_char_index[i] = brazilChars[i]


# Trying to understand whats going on here and how it works
# def decode_sequence(input_seq):
#
#     states_value=encoder_model_1.predict(input_seq)
#     target_seq = np.zeros((1, 1, len(brazilChars)))
#     target_seq[0, 0, brazilDict["\t"]] = 1.0
#     flag=0
#     sent=""
#
#     while flag==0:
#         output_tokens, h, c,h1,c1 = decoder_model.predict([target_seq] + states_value)
#         sample = np.argmax(output_tokens[0, -1, :])
#         sampled_char = reverse_target_char_index[sample]
#         sent+=sampled_char
#         if sampled_char == "\n" or len(sent) > max_decoder_seq_length:
#             flag=1
#         target_seq = np.zeros((1, 1, len(brazilChars)))
#         target_seq[0, 0,sample] = 1.0
#         states_value = [h, c,h1,c1]
#     return sent
#
#
#
#
#
#
#
# english='Chair'
# k=len(english)
# m=0
# a=[]
# b=[]
# c=[]
# inpu=[]
# while m<k:
#     for char in english[m]:
#         for i in range(len(englishChars)):
#             if englishDict[char]==i:
#                 a.append(1)
#             else:
#                 a.append(0)
#     c.append(a)
#     a=[]
#     m=m+1
# while m<max_encoder_seq_length:
#         for i in range(len(englishChars)):
#             if i==0:
#                 a.append(1)
#             else:
#                 a.append(0)
#         c.append(a)
#         a=[]
#         m=m+1
# inpu.append(c)
# inpu=np.array(inpu)
#
# d=decode_sequence(inpu)
# print(d)




