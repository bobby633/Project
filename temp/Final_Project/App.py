

from hashlib import sha256
from flask import Flask,render_template,request,redirect,session,url_for,flash
from flask_sqlalchemy import SQLAlchemy

from passlib.hash import sha256_crypt


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


import os

import tweepy
import pickle
import pandas as pd
import nltk
from datetime import datetime,timedelta
from nltk.corpus import stopwords
from profanity_filter import ProfanityFilter 
from nltk.tokenize import word_tokenize

app = Flask(__name__)
app.secret_key = "secret_key"
app.permanent_session_lifetime = timedelta(minutes=60)
stop_words = set(stopwords.words('english'))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

#will remove tensorflow texts in terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
pf = ProfanityFilter()#Nsfw words catcher

# tweepy Credentials
consumer_key= "Q28ELSx65m8jOSYk0iaZzSye4"
consumer_secret = "bgxHF5LUNO2DQRfRyYcgJrgS2E5FxnzKdA92RMpVSFHYCQa9A9"
access_token= "1445429606111223811-xHz3KWbsQqVukL7598D2GTTFoDxVsZ"
access_token_secret = "bDxXc3cxAf62XtqsyQQYI49OTHKnxGfM4TPGNigtOb2yj"

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)

class Result_model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(100) )
    search = db.Column(db.String(280) )
    positive = db.Column(db.Integer)
    negative = db.Column(db.Integer)
    neutral = db.Column(db.Integer)
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
    email = db.Column(db.String(100), unique = True,nullable = False)
    password = db.Column(db.String(100),nullable=False)

    def __init__(self,name,email,password):
        self.name = name
        self.email = email
        self.password = password




with open('sentiment_model','rb') as f:
    model = pickle.load(f)
with open('tokenizer.pickle','rb') as f:
    tokenize = pickle.load(f)

@app.route('/')
def home():
    #nltk.download('stopwords')
    if "user" in session :
        name = session['user']
        return render_template('/home.html',name=name)
   
    name = 'guest'
    return render_template('/home.html',name=name)

@app.route('/search',methods=['GET','POST'])
def search():
    if request.method == 'POST':
       search =  request.form['search']
       amount = int(request.form['amount'])
       checked = request.form['checked']
       if  not checked:
           print("checked")
        
       user = session['user']
       if pf.is_clean(search):
        tweets = tweepy.Cursor(api.search_tweets,q=f'{search} - filter:retweets', lang="en").items(amount)
        df = pd.DataFrame([tweet.text for tweet in tweets], columns= ['tweets'] )
        
        df['tweets'] = clean_text(df["tweets"])
        df['tweets'] = df['tweets'].apply(remove_stopwords)
        #df['results'] = df['tweets'].apply(predict_class)
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
            completed = Result_model(user,search,positive,negative,neutral) 
            db.session.add(completed)
            db.session.commit() 
            return render_template('/results.html',ress = ress, search = search ,positive=positive,negative=negative,neutral=neutral,oldPositive=oldPositive,oldNegative=oldNegative,oldNeutral=oldNeutral,text = [df.to_html()])
        else:
                results = all_results[-1]
                oldPositive = results.positive
                oldPositive =  list(oldPositive)[::-1]
                oldPositive =   oldPositive[-1]
                oldNegative = results.negative
                oldNegative =  list(oldNegative)[::-1]
                oldNegative = oldNegative[-1]
                oldNeutral = results.neutral
                oldNeutral =  list(oldNeutral)[::-1]
                oldNeutral = oldNeutral[-1]

                completed = Result_model(user,search,positive,negative,neutral) 
                db.session.add(completed)
                db.session.commit() 

                return render_template('/results.html',ress = ress, search = search ,positive=positive,negative=negative,neutral=neutral,oldPositive=oldPositive,oldNegative=oldNegative,oldNeutral=oldNeutral,text = [df.to_html()])
    else:
        return render_template('/search.html')
    return render_template('/search.html')

    
def clean_text(df):
    df = df.str.lower()
    df =df.str.replace(r"(?:\@|http?\://|https?\://|www)\S+"," ")
    df =df.str.replace('[^a-zA-z0-9]',' ')
    return(df)
def remove_stopwords(df):


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
   # print(text,'The predicted sentiment is', sentiment_classes[yt[0]])
    return (sentiment_classes[yt[0]])




@app.route('/results')
def result():
    return render_template('results.html')

@app.route("/view")
def view():
    return render_template("view.html",values=User.query.all())

@app.route('/about')
def about():
    return render_template('/about.html')

@app.route('/login', methods = ['GET','POST'])
def login():
    if request.method == 'POST':
        user = request.form['name']
        password = request.form['password']
        query = User.query.filter_by(name=user, password=password)
        if  query:
            session.permanent = True      
            session["user"] = user
            return redirect(url_for('home'))
        else:
            flash("username or password doesnt match")
            return redirect(url_for('login'))      
    else:
        if "user" in session :
         return redirect(url_for('home'))   
    return render_template('login.html')


@app.route('/signup', methods = ['GET','POST'])
def signup():
    if request.method == 'POST':
        session.permanent = True
        user = request.form['name']
        email = request.form['email']
        password = request.form['password']
        password = sha256_crypt.encrypt(str(password))
        session["user"] = user
        found_user = User.query.filter_by(name=user).first() 
        if found_user:
            flash( "user already exist try another name")
        else:
            usr = User(user,email,password)
            db.session.add(usr)
            db.session.commit()

        return redirect(url_for('home'))
    else:
        if "user" in session :
         return redirect(url_for('home'))   
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop("user",None)

    return redirect(url_for('home'))
    







if __name__ == "__main__":
    #when finished remove this 
    db.create_all()
    app.run(debug=True)

  