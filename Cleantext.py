import pandas as pd
import re
import emoji
from textblob import TextBlob
class CleanText:
    def cleanText(text):
        text = re.sub(r'@[A-Za-z0-9]+',' ',text) # removes mentions
        text = re.sub(r'#',' ',text) # removes '#'
        text = re.sub(r'RT[\s]+', ' ',text) #removes retweets
        text = re.sub(r'https?:\/\/S+',' ',text)
        text = re.sub(emoji.get_emoji_regexp(), r' ', text)
        return text
     
class SubandPol:        
        #Create function to get subjecttivity
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    #create function to get the polatiorty
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity
class Analysis:
    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score==0:
            return 'Neutral'
        else:
            return 'Positive'
