import pandas as pd
import numpy as np

# misc

import matplotlib.pyplot as plt

# reddit crawler
import praw

# sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
class Reddit:
    def get_reddit():
        reddit = praw.Reddit(user_agent='Beginning_Work_7197',
                    client_id='IulmyBxvFEpbk-2LEX-Ybw',
                    client_secret='TxO5Ctn-W9gu_5xBEw83BKQp1gKQvw',
                    check_for_async=False)

        headlines = set()
        for sub in reddit.subreddit('politics').new(limit=20):
        #the more there is the slower the load times so limit will be for now 20
            headlines.add(sub.title)


        sia = SentimentIntensityAnalyzer()
        results = []
        for line in headlines:
            scores = sia.polarity_scores(line) 
            scores['headline'] = line
            results.append(scores)
            df = pd.DataFrame.from_records(results)
        df['label'] = 0
        df.loc[df['compound'] > 0.2, 'label'] = 1
        df.loc[df['compound'] < -0.2, 'label'] = -1
        print(df)

