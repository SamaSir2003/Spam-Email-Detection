
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import re
import string

pkl_model = open("model.pkl","rb")
lr = pickle.load(pkl_model)
pkl_f_ext = open("f_ext.pkl","rb")
vectorizer = pickle.load(pkl_f_ext)

def preprocess(text):
    text = text.lower()
    text = re.sub('[.?]','',text)
    text = re.sub('\W'," ",text)
    text = re.sub("https?://\S+|www.\S+",'',text)
    text = re.sub('<.?>+','',text)
    text = re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w\d\w','',text)
    return text

def prediction(n):
    if n == 1:
        return "Not a spam"
    elif n == 0:
        return "Spam Mail"

def spam_detection(message):
    testing_message = {"text":[message]}
    new_def_test = pd.DataFrame(testing_message)
    new_def_test['text'] = new_def_test['text'].apply(preprocess)
    x_test = new_def_test['text']
    new_xv_test = vectorizer.transform(x_test)
    pred_vc = lr.predict(new_xv_test)
    return ('The prediction is : {}'.format(prediction(pred_vc[0])))

def main():
    st.title(":orange[Fake message prediction]")
    st.divider()
    with st.form(key="message_form",clear_on_submit=True):
        message = st.text_area("message_Input",placeholder="Enter message",height=300,label_visibility="hidden")
        submit_button = st.form_submit_button("Predict")
    result=""
    st.divider()
    col1,col2 = st.columns(2)
    if submit_button:
        result = spam_detection(message)
        st.success(result)
    if col2.button("Reset"):
        message.replace(message,"Paste message here")
if __name__ == '__main__':
    main()