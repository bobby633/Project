import tweepy 
import datetime
import twitter_cred
import pandas as pd
import numpy as np
from Cleantext import CleanText,SubandPol,Analysis
from Graph import Graph
import pyrebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from profanity_filter import ProfanityFilter 
from SarcasticDetector import Sarcasm

pf = ProfanityFilter()#Nsfw words catcher

firebaseConfig ={
    'apiKey': "AIzaSyD_QwEW6ivMI4TEjKxTFy2r7SPa8wdtz6I",
    'authDomain': "sentimentanalysis-333813.firebaseapp.com",
    'projectId': "sentimentanalysis-333813",
    'databaseURL': 'https://sentimentanalysis-333813-default-rtdb.firebaseio.com/',
    'storageBucket': "sentimentanalysis-333813.appspot.com",
    'messagingSenderId': "487751248294",
    'appId': "1:487751248294:web:810d39d332013c4ab0c767",
    'measurementId': "G-0B2YHS542J"
}
firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()




auth = tweepy.AppAuthHandler(twitter_cred.consumer_key, twitter_cred.consumer_secret)
api = tweepy.API(auth)
class Twitter:
    def twitter():
        print (f'What Word?')  
        
        search_word = input()
        if pf.is_clean(search_word):
            start_date = "2021-11-1"
            post = tweepy.Cursor(api.search_tweets, result_type = 'popular' ,q=search_word,lang="en",tweet_mode="extended").items(10) 
            #Create a DF 
        else:
            print ("not a nice word")
            return
        df = pd.DataFrame([tweet.full_text for tweet in post], columns=['Tweets'])
        #Remove special Char

        df['Tweets'] = df['Tweets'].apply(CleanText.cleanText)

            
        df['Tweets'].apply(Sarcasm.predict_sarcasm)
        #create new columns
        #print(df['Tweets'])
        df['subjectivity']= df['Tweets'].apply(SubandPol.getSubjectivity)
        df['polarity']=df['Tweets'].apply(SubandPol.getPolarity)
       # data = {"SearchWord":search_word,"Polarity":df['polarity'].to_json()}
        #db.child("SearchWords")
        #data = {"Word":search_word}
        #db.child("Words").push(data)
        #ref = db.child("SearchWords").get()
        #print(ref.val())
        
        #Positive
        df['Analysis'] = df['polarity'].apply(Analysis.getAnalysis)
        positive = df[df.Analysis == 'Positive']
        positive = positive['Tweets']
        positvePercentage = round( (positive.shape[0] / df.shape[0]) *100 , 1)
        print(positvePercentage , "Positve")
        #Send Positive
        #db.child("Percentage")
        #data = {"positive":positvePercentage}
        #db.child("positive").push(data)
        #ref = db.child("Percentage").get()
        
        #Negative
        df['Analysis'] = df['polarity'].apply(Analysis.getAnalysis)
        negative = df[df.Analysis == 'Negative']
        negative = negative['Tweets']
        negativePercentage = round( (negative.shape[0] / df.shape[0]) *100 , 1)
        print(negativePercentage , "Negative")
        #Send Negative
        #db.child("Percentage")
        #data = {"negative":negativePercentage}
        #db.child("negative").push(data)
        #ref = db.child("Percentage").get()
        
        #Neutral
        df['Analysis'] = df['polarity'].apply(Analysis.getAnalysis)
        neutral = df[df.Analysis == 'Neutral']
        neutral = neutral['Tweets']
        neutralPercentage = round( (neutral.shape[0] / df.shape[0]) *100 , 1)
        print(neutralPercentage , "Neutral")
        #GetNeutral
        #db.child("Percentage")
        #data = {"neutral":neutralPercentage}
        #db.child("neutral").push(data)
        ref = db.child("Percentage").get()
       # for old in ref.each():
       #    print(old.val())
        
        #print(ref.val())

        Graph.graph(df,'Twitter')
    def retweets():
        print (f'What user?')  
 #       count = 20
 #       search_word = input()
 #       if pf.is_clean(search_word):
 #           start_date = "2021-11-1"
 #           retweets_list = api.retweets(search_word,count)
 #           for retweet in retweets_list:
 #               print(retweet.user.screen_name)
            #Create a DF 
       
###########################  Twitter__________Users           #######################  
class TwitterUser:
    def twitter_user():
        user ="BillGates"
        start_date = "2021-11-1"
        post = api.user_timeline(screen_name = user, count=100,tweet_mode="extended")
        #Create a DF 
        df = pd.DataFrame([tweet.full_text for tweet in post], columns=['Tweets'])
        #Remove special Char
        start = datetime.datetime.now()
        df['Tweets'] = df['Tweets'].apply(CleanText.cleanText)
        finish = datetime.datetime.now()
        print(finish-start)
        #create new columns
        start = datetime.datetime.now()
        df['subjectivity']= df['Tweets'].apply(SubandPol.getSubjectivity)
        df['polarity']=df['Tweets'].apply(SubandPol.getPolarity)
        df['Analysis'] = df['polarity'].apply(Analysis.getAnalysis)
        finish = datetime.datetime.now()
        print(finish-start)
        Graph.graph(df,'Twitter')
