""" @package FYP
    Description
    the Sentiment Analysis of social media accounts, 
    this give the user a clear understand of how the emotion of a 
    certain word or a certin persons in a clean and organised manner  
    @author Bobby Mitchell
"""
from flask import Flask,render_template,request,redirect,session,url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime,timedelta
from nltk.corpus import stopwords
import nltk
import tweepy
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os

import waitress


##if stopwards arent downloaded already
##uncomment next line once and run and install
#nltk.download('stopwords') 








app = Flask(__name__)
app.secret_key = "secret_key"
app.permanent_session_lifetime = timedelta(minutes=60)
stop_words = set(stopwords.words('english'))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)




# tweepy Credentials usually would be hidden but this is only a college project
consumer_key= "Q28ELSx65m8jOSYk0iaZzSye4"
consumer_secret = "bgxHF5LUNO2DQRfRyYcgJrgS2E5FxnzKdA92RMpVSFHYCQa9A9"
access_token= "1445429606111223811-xHz3KWbsQqVukL7598D2GTTFoDxVsZ"
access_token_secret = "bDxXc3cxAf62XtqsyQQYI49OTHKnxGfM4TPGNigtOb2yj"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)



tf.saved_model.LoadOptions(
    allow_partial_checkpoint=False,
    experimental_io_device=None,
    experimental_skip_checkpoint=False
)


"""
the class models : 
Result_model is where the results are stored  alongside the username to keep
track of who looked up what word 

User model is where the user information is kept
"""
class Result_model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(100) )
    search = db.Column(db.String(280) )
    positive = db.Column(db.String(20))
    negative = db.Column(db.String(20))
    neutral = db.Column(db.String(20))
    searched_at = db.Column(db.DateTime,default=datetime.utcnow)
    def __init__(self,user,search,positive,negative,neutral):
        self.user = user
        self.search = search
        self.positive = positive
        self.negative = negative
        self.neutral = neutral

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100),unique = True,nullable = False)
    password = db.Column(db.String(100),nullable=False)

    def __init__(self,name,password):
        self.name = name
        self.password = password


# Lstm Sentiment model and token file
model = keras.models.load_model('new_Sentiment_model.h5')



with open('tokenizer.pickle','rb') as t:
    tokenize = pickle.load(t)







@app.route('/')
def home():
    """
    home page if the session user is not signed in it will redirect to the
    to the login page 
    More details. 
    if user is signed in their username is displayed on homepage
    """
    if "user" in session :
        name = session['user']
        return render_template('/home.html',name=name)
    else:
        return redirect(url_for('login'))

@app.route('/search',methods=['GET','POST'])
def search():
    """
    Searches for a tweets in tweepy
    the tweets is then processed by cleaning and removing any stopwords
    the tweets is then used in the prediction LSTM Algorithm for Sentiment Analysis 
    the previous version of the same searched word is also displayed 
    """
    if request.method == 'POST':
        search =  request.form['search']
        amount = int(request.form['amount'])
        
        user = session['user']
    
        tweets = tweepy.Cursor(api.search_tweets,q=f'{search} - filter:retweets', lang="en").items(amount)
        df = pd.DataFrame([tweet.text for tweet in tweets], columns= ['tweets'] )
        #preprocessed tweets are cleaned
        df['tweets'] = clean_text(df["tweets"])
        df['tweets'] = df['tweets'].apply(remove_stopwords)
       
        #the sentiment Analysis prediction 
        df['sentiment'] = df['tweets'].apply(lambda tweets : predict_class([tweets]))
       
        #seperating the data into new data
        positive = df.loc[df['sentiment'] == 'Positive'].count()[0]
        negative = df.loc[df['sentiment'] == 'Negative'].count()[0]
        neutral = df.loc[df['sentiment'] == 'Neutral'].count()[0]

        #most recent data before updated
        all_results = Result_model.query.filter_by(user = user,search = search).all()
        results = all_results
        #if this word hasnt been used before fill with empty values
        if not results :
            oldNegative = 0
            oldNeutral = 0
            oldPositive = 0
            #the searched word is sent to the database
            #integers are made into 'str' because when viewing ints on 
            #sqlAlchemy they return ints as bytes 
            completed = Result_model(user,search,str(positive),str(negative),str(neutral)) 
            db.session.add(completed)
            db.session.commit()
            #the variables are sent to the 
            # results page 
            return render_template('/results.html', search = search ,positive=positive,negative=negative,neutral=neutral,oldPositive=oldPositive,oldNegative=oldNegative,oldNeutral=oldNeutral,text = [df.to_html()])
       
        #if the word has been viewed before 
        else:
            #from the database the latest result is retrieved 
            #the sentiment analysis of the searched word is added to the variables below
            #the are changed from string to int
            results = all_results[-1]
            oldPositive = int(results.positive)                                
            oldNegative = int(results.negative)
            oldNeutral = int(results.neutral)
            completed = Result_model(user,search,str(positive),str(negative),str(neutral)) 
            db.session.add(completed)
            db.session.commit() 

            return render_template('/results.html', search = search ,positive=positive,negative=negative,neutral=neutral,oldPositive=oldPositive,oldNegative=oldNegative,oldNeutral=oldNeutral,text = [df.to_html()])
    else:
        return render_template('/search.html')
    

@app.route('/search-user',methods=['GET','POST'])

def search_user():
    """
    the "search_user" is the exact same def as "search" def above 
    except a specific user tweets are retieved
    """
    if request.method == 'POST':
        search =  request.form['search']
        amount = int(request.form['amount'])       
        user = session['user']
        post = api.user_timeline(screen_name = search,count=amount,tweet_mode="extended")
        #Create a DF 
        df = pd.DataFrame([tweet.full_text for tweet in post], columns=['tweets'])
        #preprocessed tweets are cleaned
        df['tweets'] = clean_text(df["tweets"])
        df['tweets'] = df['tweets'].apply(remove_stopwords)
        #the sentiment Analysis prediction 
        df['sentiment'] = df['tweets'].apply(lambda tweets : predict_class([tweets]))
        #seperating the data into new data
        positive = df.loc[df['sentiment'] == 'Positive'].count()[0]
        negative = df.loc[df['sentiment'] == 'Negative'].count()[0]
        neutral = df.loc[df['sentiment'] == 'Neutral'].count()[0]

        #most recent data before updated
        all_results = Result_model.query.filter_by(user = user,search = search).all()
        user_results = Result_model.query.filter_by(user = user,search = search).all()
        ress = user_results
        results = all_results
        if not results :
            oldNegative = 0
            oldNeutral = 0
            oldPositive = 0
            completed = Result_model(user,search,str(positive),str(negative),str(neutral)) 
            db.session.add(completed)
            db.session.commit() 
            return render_template('/results.html',ress = ress, search = search ,positive=positive,negative=negative,neutral=neutral,oldPositive=oldPositive,oldNegative=oldNegative,oldNeutral=oldNeutral,text = [df.to_html()])
        else:
            results = all_results[-1]
            oldPositive = int(results.positive)          
            oldNegative = int(results.negative)
            oldNeutral = int(results.neutral)
            completed = Result_model(user,search,str(positive),str(negative),str(neutral)) 
            db.session.add(completed)
            db.session.commit() 

            return render_template('/results.html',ress = ress, search = search ,positive=positive,negative=negative,neutral=neutral,oldPositive=oldPositive,oldNegative=oldNegative,oldNeutral=oldNeutral,text = [df.to_html()])
    else:
        return render_template('/search-user.html')



@app.route('/history',methods=['GET','POST'])
def history():
    """
    if the user wants to view old searched up results 
    they can look them
    the user is greeted with a statistcs of that searched word 
    everytime its been searched
    """
    if request.method == 'POST':
        if "user" in session:
            search =  request.form['search']
            user = session['user']
            
            return render_template('history.html',values = Result_model.query.filter_by(user = user,search = search).all())
        else:
            return redirect(url_for('login'))
    return render_template('history.html')
@app.route('/results')
def result():
    """
    displays the results from searching a user or a term 
    """
    return render_template('results.html')


@app.route('/about')
def about():
    """
    the about page
    """
    return render_template('/about.html')

@app.route('/login', methods = ['GET','POST'])
def login():
    """
    using SQLAlchemy as the database the user can login if the user has an account
    if the user does not have an account the user will not log in
    this is the default page if the user is not logged in
    a sign up option is present on the page
    """
    if request.method == 'POST':
        user = request.form['name']
        password = request.form['password']
        query = User.query.filter_by(name=user,password=password).first() 
        #if this username and password is correct the can sign in 
        if  query:
            session.permanent = True      
            session["user"] = user
            return redirect(url_for('home'))

        else:
            #otherwise try again
            warning = "error try again"
            return render_template('login.html',warning=warning)      
    else:
        #if the user is in session they should not be able to log in again
        #redirects the user to the home page instead
        if "user" in session :
         return redirect(url_for('home'))   
    return render_template('login.html')


@app.route('/signup', methods = ['GET','POST'])
def signup():
    """
    if the user doesnt have an account they can make one
    it needs to be a unqiue name
    password fits the password credentials
    """
    if request.method == 'POST':
        session.permanent = True
        user = request.form['name']
        password = request.form['password']
        session["user"] = user
        usr = User(user,password)
        db.session.add(usr)
        db.session.commit()
        return redirect(url_for('home'))
    else:
        if "user" in session :
         return redirect(url_for('home'))   
    return render_template('signup.html')

@app.route('/logout')
def logout():
    """
    logs the user out and removes their session
    """
    session.pop("user",None)
    session.pop("password",None)
    return redirect(url_for('home'))
    



def clean_text(df):
    """
    removes special characters, makes the tweets lowercase
    """
    df = df.str.lower()
    df =df.str.replace(r"(?:\@|http?\://|https?\://|www)\S+"," ")
    df =df.str.replace('[^a-zA-z0-9]',' ')
    return(df)
def remove_stopwords(df):
    """
    stop words are removed
    stop words are words that have no value to the sentiment analysis
    eg. "'a' ,'i' ,'the' ,'it' "
    """

    word_list=df.split()
    df=' '.join([word for word in word_list if word.lower() not in stop_words])
    
    return(df)

max_words = 5000
max_len=50

def tokenize_pad_sequences(df):
    '''
    This function tokenize the input text into sequnences of intergers and then
    pad each sequence to the same length
    '''
    # Text tokenization
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(df['tweets'])
    # Transforms text to a sequence of integers
    df['token'] = tokenizer.texts_to_sequences(df['tweets'].values)
    # Pad sequences to the same length
    df['token'] = pad_sequences(df['token'], padding='post', maxlen=max_len).tolist()
    # return sequences
    return df['token']

def predict_class(text):
    '''Function to predict sentiment class of the passed text'''
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len=50 
    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenize.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    # Print the predicted sentiment

    return (sentiment_classes[yt[0]])








if __name__ == "__main__":
    port = int(os.eviron.get('PORT',33507))    
    db.create_all()
    waitress.server(app,port=port)