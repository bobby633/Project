#Sentiment Analysis Program from twitter
#This will get the tweets
import tweepy
import sys

import twitter_cred


# Consumer keys and access tokens, used for OAuth
auth = tweepy.AppAuthHandler(twitter_cred.consumer_key, twitter_cred.consumer_secret)
class Print_Tweets():
    def stream_tweets(self,hashtags):
        for tweet in tweepy.Cursor(api.search_tweets, q=hashtags).items(20):
         #prints the emojis on the json txt
            with open("tweets.json", 'a') as tf:
                tf.write(tweet.text)                   
                print(tweet.text)
                print("")
                print("------------")
                print("") 
        tf.close() 
        return True
        
            
if __name__== "__main__":
    api = tweepy.API(auth)
    hashtags = ["$shib"]
    tweet_streamer = Print_Tweets()
    tweet_streamer.stream_tweets(hashtags)