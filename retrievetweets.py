import tweepy 
from textblob import TextBlob
import emoji
import twitter_cred
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

class twitter:
    def __init__(self):
        return self
    #Gets credentials
    auth = tweepy.AppAuthHandler(twitter_cred.consumer_key, twitter_cred.consumer_secret)
    api = tweepy.API(auth)
    
    search_word ="love"
    start_date = "2021-11-1"
    post = tweepy.Cursor(api.search_tweets, q=search_word,lang="en",tweet_mode="extended").items(50)
    i = 1
    
    #Create a DF 
    df = pd.DataFrame([tweet.full_text for tweet in post], columns=['Tweets'])
    
    #Remove special Char
    def cleanText(text):
        text = re.sub(r'@[A-Za-z0-9]+',' ',text) # removes mentions
        text = re.sub(r'#',' ',text) # removes '#'
        text = re.sub(r'RT[\s]+', ' ',text) #removes retweets
        text = re.sub(r'https?:\/\/S+',' ',text)
        text = re.sub(emoji.get_emoji_regexp(), r"", text)
        return text
    df['Tweets'] = df['Tweets'].apply(cleanText)
    
    #Create function to get subjecttivity
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity
    
    #create function to get the polatiorty
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity
    
    #create new columns
    df['subjectivity']= df['Tweets'].apply(getSubjectivity)
    df['polarity']=df['Tweets'].apply(getPolarity)
    
    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score==0:
            return 'Neutral'
        else:
            return 'Positive'
    
    df['Analysis'] = df['polarity'].apply(getAnalysis)
    print(df)
    
    #do an if statement if they want barchart or pie or line
    if __name__== "__main__":
        print("bar scatter,text or pie")
        Choice = input()
        if Choice == 'scatter':
            plt.figure(figsize =(5,3))
            for i in range(0,df.shape[0]):
                plt.scatter(df['polarity'][i],df['subjectivity'][i], color = 'Green')
            plt.title('Sentmient analysis')
            plt.xlabel('polarity')
            plt.ylabel('subjectivity')
            plt.show()
        elif Choice == 'bar':
        
            df['Analysis'].value_counts()
            #plot and visuals the counts
            plt.title('sentiment analysis')
            plt.xlabel('sentiment')
            plt.ylabel('Counts')
            df['Analysis'].value_counts().plot(kind='bar')
            plt.show()
        elif Choice == 'pie':
            df['Analysis'].value_counts()
        
            #plot and visuals the counts
            plt.title('sentiment analysis')
            df['Analysis'].value_counts().plot(kind='pie')
            plt.show()
        elif Choice == 'text':
            print(df)
        else:
            print('Not a choice')