import matplotlib.pyplot as plt
import pandas as pd
import datetime
class Graph:
    def graph(df):
        
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
            Graph.graph
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
            start = datetime.datetime.now()
            print(df)
            finish = datetime.datetime.now()
            print(finish-start)
        else:
            print('Not a choice')            
