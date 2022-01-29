import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras import losses
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

def model():
    train_string=r"C:\Users\DELL LATITUDE\Desktop\Manthan\Data set\LIAR dataset\Liar_train_true_false_PreProcessed.csv"
    train_data=pd.read_csv(train_string)
    test_string=r"C:\Users\DELL LATITUDE\Desktop\Manthan\Data set\LIAR dataset\Liar_test_true_false_PreProcessed.csv"
    test_data=pd.read_csv(test_string)
    print(train_data.head(2))
    print(test_data.head(2))
    train_data.dropna(axis = 0, how ='any',inplace=True) 
    train_data['Num_words_text'] = train_data['PreProcessed_News'].apply(lambda x:len(str(x).split())) 
    mask = train_data['Num_words_text'] >2
    train_data = train_data[mask]
    print('-------Train data--------')
    print(train_data['Label'].value_counts())
    print(len(train_data))
    print('-------------------------')
    max_train_sentence_length  = train_data['Num_words_text'].max()
    print('Train Max Sentence Length :'+str(max_train_sentence_length))
    test_data.dropna(axis = 0, how ='any',inplace=True) 

    test_data['Num_words_text'] = test_data['PreProcessed_News'].apply(lambda x:len(str(x).split())) 
    max_test_sentence_length  = test_data['Num_words_text'].max()

    mask = test_data['Num_words_text'] >2
    test_data = test_data[mask]

    print('-------Test data--------')
    print(test_data['Label'].value_counts())
    print(len(test_data))
    print('-------------------------')

    print('Test Max Sentence Length :'+str(max_test_sentence_length))

    num_words = 20000

    tokenizer = Tokenizer(num_words=num_words,oov_token="unk")
    tokenizer.fit_on_texts(train_data['PreProcessed_News'].tolist())


    print(str(tokenizer.texts_to_sequences(['john died'])))
    X_train, X_valid, y_train, y_valid = train_test_split(train_data['PreProcessed_News'].tolist(),\
                                                      train_data['Label'].tolist(),\
                                                      test_size=0.1,\
                                                      stratify = train_data['Label'].tolist(),\
                                                      random_state=0)


    print('Train data len:'+str(len(X_train)))
    print('Class distribution'+str(Counter(y_train)))
    print('Valid data len:'+str(len(X_valid)))
    print('Class distribution'+ str(Counter(y_valid)))


    x_train = np.array( tokenizer.texts_to_sequences(X_train) )
    x_valid = np.array( tokenizer.texts_to_sequences(X_valid) )
    x_test  = np.array( tokenizer.texts_to_sequences(test_data['PreProcessed_News'].tolist()) )



    x_train = pad_sequences(x_train, padding='post', maxlen=400)
    x_valid = pad_sequences(x_valid, padding='post', maxlen=400)
    x_test = pad_sequences(x_test, padding='post', maxlen=400)

    print(x_train[0])

    le = LabelEncoder()

    train_labels = le.fit_transform(y_train)
    train_labels = np.asarray( tf.keras.utils.to_categorical(train_labels))
    #print(train_labels)
    valid_labels = le.transform(y_valid)
    valid_labels = np.asarray( tf.keras.utils.to_categorical(valid_labels))

    test_labels = le.transform(test_data['Label'].tolist())
    test_labels = np.asarray(tf.keras.utils.to_categorical(test_labels))
    list(le.classes_)


    train_ds = tf.data.Dataset.from_tensor_slices((x_train,train_labels))
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid,valid_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test,test_labels))
    print(y_train[:10])
    train_labels = le.fit_transform(y_train)
    print('Text to number')
    print(train_labels[:10])
    train_labels = np.asarray( tf.keras.utils.to_categorical(train_labels))
    print('Number to category')
    print(train_labels[:10])
    count =0
    print('======Train dataset ====')
    for value,label in train_ds:
        count += 1
        print(value,label)
        if count==3:
            break
    count =0
    print('======Validation dataset ====')
    for value,label in valid_ds:
        count += 1
        print(value,label)
        if count==3:
            break
    print('======Test dataset ====')
    for value,label in test_ds:
        count += 1
        print(value,label)
        if count==3:
            break  
    sample = open(r"C:\Users\DELL LATITUDE\Desktop\Manthan\Data set\LIAR dataset\inpLiar.txt", encoding="utf8")
    s = sample.read()

    # Replaces escape character with space
    f = s.replace("\n", " ")

    data = []

    # iterate through each sentence in the file
    for i in sent_tokenize(f):
        temp = []
	
	    # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())

        data.append(temp)

# Create CBOW model
    model1 = gensim.models.Word2Vec(data, min_count = 1,vector_size = 4,window = 1)

# Create Skip Gram model
    model2 = gensim.models.Word2Vec(data, min_count = 1,vector_size = 4,window = 1, sg = 1)

    print(model1.wv['gun'])
    l=[]
    for i in range(len(data[0])):
        a=model1.wv[data[0][i]]
        #print(a)
        l.append(a)
    print(len(l))   
    weightMat=np.array(l)
    w=weightMat[:20001]
    #w.shape=(20001, 4)
    max_features =20000
    embedding_dim =4
    sequence_length = 400

    model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Embedding(max_features +1, embedding_dim,weights=[l] ,input_length=sequence_length,\
    #                                   embeddings_regularizer = regularizers.l2(0.0005)))                                    

    model.add(tf.keras.layers.Embedding(max_features +1, embedding_dim,weights=[w],input_length=sequence_length,\
                                    embeddings_regularizer = regularizers.l2(0.0005)))                                    

    model.add(tf.keras.layers.Conv1D(128,2, activation='relu',\
                                 kernel_regularizer = regularizers.l2(0.0005),\
                                 bias_regularizer = regularizers.l2(0.0005)))                               


    model.add(tf.keras.layers.GlobalMaxPooling1D())

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(2, activation='sigmoid',\
                                kernel_regularizer=regularizers.l2(0.001),\
                                bias_regularizer=regularizers.l2(0.001),))
                               



    print(model.summary())
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='Nadam', metrics=["CategoricalAccuracy"])
    epochs = 10
    # Fit the model using the train and test datasets.
    #history = model.fit(x_train, train_labels,validation_data= (x_test,test_labels),epochs=epochs )
    history = model.fit(train_ds.shuffle(2000).batch(128),
                    epochs= epochs ,
                    validation_data=valid_ds.batch(128),
                    verbose=1)

    print(history.history)
    plt.plot(history.history['loss'], label=' training data')
    plt.plot(history.history['val_loss'], label='validation data)')
    plt.title('Loss for Text Classification')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(history.history['categorical_accuracy'], label=' (training data)')
    plt.plot(history.history['val_categorical_accuracy'], label='CategoricalCrossentropy (validation data)')
    plt.title('CategoricalAccuracy for Text Classification')
    plt.ylabel('CategoricalAccuracy value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()
    x_test  = np.array( tokenizer.texts_to_sequences(test_data['PreProcessed_News'].tolist()) )
    x_test  = np.array( tokenizer.texts_to_sequences(test_data['PreProcessed_News'].tolist()) )
    x_test = pad_sequences(x_test, padding='post', maxlen=400)
    print("Generate predictions for all samples")
    predictions = model.predict(x_test)
    print(predictions)
    predict_results = predictions.argmax(axis=1)
    test_data['pred_Label']= predict_results
    test_data['pred_Label'] = np.where((test_data.pred_Label == 0),'False',test_data.pred_Label)
    test_data['pred_Label'] = np.where((test_data.pred_Label == '1'),'True',test_data.pred_Label)
    #test_data["Label"].shape=(1258,)=test_data["pred_Label"].shape
    labels = ['False', 'True']
    
    print(classification_report(test_data['Label'].astype(str).tolist(),test_data['pred_Label'].astype(str).tolist(),labels=labels))
    
    #saving the model
    model.save(r'C:\Users\DELL LATITUDE\Desktop\Manthan\Kavya programs\Frontend-flask\model.h5')
def preprocessTest():
    df=pd.read_csv("output.csv")
    raw_text=df['News']
    print(raw_text.head(2))
    #Convert into lower text
    clean_text_1=[]
    def toLower(data):
        for words in raw_text:
            clean_text_1.append(str.lower(words))
    toLower(raw_text)
    #Stage 2 -- tokenize
    clear_text_2=[]
    sent_tok=[]
    for sent in clean_text_1:
        sent=sent_tokenize(sent)
        sent_tok.append(sent)
    # Word tokenize
    clean_text_2=[ word_tokenize(i) for i in clean_text_1]
    #removing punctuations

    clean_text_3=[]

    for words in clean_text_2:
        clean=[]
        for w in words:
            res=re.sub(r'[^\w\s]' , "", w)
            if res!="":
                clean.append(res)
        clean_text_3.append(clean)
    #Stage 4
    #Removing Stop Words -- repeated words
    clean_text_4=[]

    for words in clean_text_3:
        w=[]
        for word in words:
            if not word in stopwords.words('english'):
                w.append(word)
        clean_text_4.append(w)
    port=PorterStemmer()
    clean_text5=[]

    for words in clean_text_4:
        w=[]
        for word in words:
            w.append(word)
        clean_text5.append(w)
    wnet=WordNetLemmatizer()
    lem=[]

    for words in clean_text_4:
        w=[]
        for word in words:
            w.append(wnet.lemmatize(word))
        lem.append(" ".join(w))
    df['PreProcessed_News']=lem
    df.to_csv('output_preprocessed.csv')        
        

    

def testing():
    from tensorflow.keras.preprocessing.text import Tokenizer
       #run this for 1st time alone else comment it
    #model()
    from tensorflow.keras.models import load_model
    new_model = load_model('model.h5')
    #print("new_model is called")
    #print(new_model.summary())
    #for single test data do only once

    preprocessTest()
    testing=pd.read_csv("output_preprocessed.csv")
    tokenizer = Tokenizer(num_words=20000,oov_token="unk")
    tokenizer.fit_on_texts(testing['PreProcessed_News'].tolist())

    z_test  = np.array( tokenizer.texts_to_sequences(testing['PreProcessed_News'].tolist()) )
    z_test = pad_sequences(z_test, padding='post', maxlen=400)
    #print("Generate predictions for all samples")
    predictions = new_model.predict(z_test)
    #print(predictions)
    predict_results = predictions.argmax(axis=1)
    testing['pred_Label']= predict_results
    #print(testing.head(2))
    testing['pred_Label'] = np.where((testing.pred_Label == 0),"False",testing.pred_Label)
    testing['pred_Label'] = np.where((testing.pred_Label == '1'),"True",testing.pred_Label)
    testing.to_csv('result.csv')        
        
    #print(testing.head(5))
    #result=[]
    #fake news are displayed as list
    res=[] 
    count=[]
    
    for x in range(0,len(testing['pred_Label'])):
        if (testing['pred_Label'][x]=="False"):
            #count=count+1
            res.append(testing['News'][x])
            count.append(testing['pred_Label'][x])
    print(len(res))
    print("--=--=7687654-================")
    print(res)
    na=pd.DataFrame(res)
    na.to_csv("false.csv")
    #print(result)
testing()
 
#for n in result:        
 #       print(n)

 