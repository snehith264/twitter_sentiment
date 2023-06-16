import pickle
import streamlit as st
import sklearn
import nltk
nltk.download('punkt')
import emoji
st.title(":red[Twitter sentiment analysis] :sunglasses:")
st.subheader('You can enter any tweet and it will predict if the tweet is Positive or Negative or Neutral or Irrelevant')
tweet_input=st.text_input("Enter Your Tweet :")
model=pickle.load(open('logistic_regression_clf.pkl', "rb"))
vectorizer=pickle.load(open('bag_of_words.pkl',"rb"))
if st.button('Click here to analyse'):
    st.spinner(text="In progress...")
    st.balloons()
    st.snow()
    bag_of_words = vectorizer.transform([tweet_input])
    sentiment=model.predict(bag_of_words)
    if(sentiment=='Positive'):
        smile = emoji.emojize(":smiley:")
        st.write(smile)
        st.write('Positive')
    elif(sentiment=='Negative'):
        disapp = emoji.emojize(":disappointed:")
        st.write(disapp)
        st.write('Negative')
    elif(sentiment=='Neutral'):
        disapp = emoji.emojize(":neutral_face:")
        st.write(disapp)
        st.write('Neutral')
    else:
        disapp = emoji.emojize(":no_mouth:")
        st.write(disapp)
        st.write('Neutral')