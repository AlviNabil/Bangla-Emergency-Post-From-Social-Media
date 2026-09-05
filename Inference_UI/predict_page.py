import streamlit as st
import numpy as np
import keras
from tokenizer import tokenize, transform, tokenize2
import tensorflow as tf
import joblib
from bert_embedding import embedding
from model_store import model_file, model_dir

# fall back to CPU when no GPU is visible to TensorFlow (needs the CUDA runtime,
# not just an NVIDIA card) - hardcoding 'GPU:0' raises on a CPU-only install
DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

# Models are fetched from the Hugging Face Hub on first use and cached under
# ~/.cache/huggingface. Without these, every Predict click re-downloaded or
# re-read a 600 MB-1 GB file.

@st.cache_resource(show_spinner=False)
def load_keras(rel_path):
    with tf.device(DEVICE):
        return keras.models.load_model(model_dir(rel_path))


@st.cache_resource(show_spinner=False)
def load_sklearn(rel_path):
    return joblib.load(model_file(rel_path))


FIRST_RUN_NOTE = ("Loading the model. The first use of each model downloads it "
                  "(up to ~1 GB) and can take several minutes.")

# def load_model():
    

def show_predict_page():
    st.title("EMEREGENCY POST PREDICTION ")
    

    Methodology = (
        "Traditional Machine Learning Methods",
        "Deep Neural Network Methods",
        "Transformer Based Methods"
    )

    classes = {0:'Accident', 1:'Blood', 2:'Crime', 3:'Fire',4:'Natural Disaster', 5:'Pandemic', 6:'Suicide', 7:'War', 8:'Weather'}
    LABEL_TO_ID = {'accident':0, 'blood':1, 'crime':2, 'fire':3, 'natural_disaster':4,
                   'pandemic':5, 'suicide':6, 'war':7, 'weather':8}


    ML_ModelNames=("Logistic Regression", "Multinomial Naive Bayes", "KNN")
    DNN_ModelNames=("LSTM", "BiLSTM", "BiLSTM+CNN")
    Transformers=("mBERT","BanglaBERT","XLM-RoBERTa")

    Types_of_Approach = st.sidebar.selectbox("Approach", Methodology)
    if(Types_of_Approach=="Traditional Machine Learning Methods"):
        Model_name=st.selectbox("Model Name", ML_ModelNames)
    elif(Types_of_Approach=="Deep Neural Network Methods"):
        Model_name=st.selectbox("Model Name", DNN_ModelNames)
    else:
        Model_name=st.selectbox("Model Name", Transformers)
    text = st.text_input("Emergency Post Classification")
    ok = st.button("Predict")
    if ok:

       
        text = [text]
        # st.write(text)
        x = -1
        if(Model_name=="BiLSTM+CNN"):
            padded_text = tokenize2(text)
            
            with st.spinner(FIRST_RUN_NOTE):
                model = load_keras('dnn/BiLSTM_CNN.model')

            predict_class=model.predict(padded_text) 
            classes_x=np.argmax(predict_class,axis=1)
            
        elif(Model_name=="BiLSTM"):
            padded_text = tokenize(text)
            with st.spinner(FIRST_RUN_NOTE):
                model = load_keras('dnn/BiLSTM.model')
            predict_class=model.predict(padded_text) 
            classes_x=np.argmax(predict_class,axis=1)
        elif(Model_name=="LSTM"):
            padded_text = tokenize2(text)   # LSTM.model expects length 600
            with st.spinner(FIRST_RUN_NOTE):
                model = load_keras('dnn/LSTM.model')
            predict_class=model.predict(padded_text) 
            classes_x=np.argmax(predict_class,axis=1)
        

        elif(Model_name=="Logistic Regression"):
            transformed_text = transform(text)
            with st.spinner(FIRST_RUN_NOTE):
                loaded_model = load_sklearn('machine_learning/logistic/finalized_model.pkl')
            classes_x=loaded_model.predict(transformed_text)
            
        elif(Model_name=="KNN"):
            transformed_text = transform(text)
            with st.spinner(FIRST_RUN_NOTE):
                loaded_model = load_sklearn('machine_learning/knn/model2.pkl')
            classes_x=loaded_model.predict(transformed_text)
        
        elif(Model_name=="Multinomial Naive Bayes"):
            transformed_text = transform(text)
            with st.spinner(FIRST_RUN_NOTE):
                loaded_model = load_sklearn('machine_learning/mnb/model.pkl')
            classes_x=loaded_model.predict(transformed_text)

        elif(Model_name=='BanglaBERT'):
            with st.spinner(FIRST_RUN_NOTE):
                classes_x = embedding(text, Model_name)
        elif(Model_name=='XLM-RoBERTa'):
            with st.spinner(FIRST_RUN_NOTE):
                classes_x = embedding(text, Model_name)
        elif(Model_name=='mBERT'):
            with st.spinner(FIRST_RUN_NOTE):
                classes_x = embedding(text, Model_name)
        
        x = classes_x[0]
        # the mnb model was fitted on string labels ('crime'), while logistic/knn
        # and the DNN/transformer paths yield integer ids - normalise to an id
        if isinstance(x, str):
            x = LABEL_TO_ID[x.strip().lower().replace(' ', '_')]
        x = int(x)
        result = "The text is classified as "+ classes[x] + " emergency."
        st.subheader(f"{result}")
        
        
        if(x==4):
            st.write("Notify The Emergency Dispatch Unit.")
            st.markdown("**:red[For more info Call: 01717171717]**")
        elif(x==2):
            st.write("Notify Crime Investigation Unit.")
            st.markdown("**:blue[For more info call: 01919191919]**")
        elif(x==3):
            st.write("Notify Fire Service Unit.")
            st.markdown("**:blue[For more info call: 0181818181]**")
        elif(x==0):
            st.write("Notify Emergency Medical Support and Local Police Station and Emergency Rescuer.")
            st.markdown("**:red[For more info about Medical Support call: 0181818181]**")
            st.markdown("**:blue[To Contact with Local Police call: 0181818181]**")
            st.markdown("**:green[To Contact with Emergency Rescuer call: 0181818181]**")
        elif(x==1):
            st.write("Notify Local Blood Bank.")
            st.markdown("**:red[For more info call: 0181818181]**")
        elif(x==5):
            st.write("Notify Local Local Government and Health Organization.")
            st.markdown("**:red[For more info call: 0181818181]**")
        elif(x==6):
            st.write("Notify Local Blood Bank.")
            st.markdown("**:red[For more info call: 0181818181]**")
        elif(x==7):
            st.write("Notify Local Blood Bank.")
            st.markdown("**:red[For more info call: 0181818181]**")
        elif(x==8):
            st.write("Notify Local Blood Bank.")
            st.markdown("**:red[For more info call: 0181818181]**")