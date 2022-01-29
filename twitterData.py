import datetime
from datetime import date
from config import *
import tweepy
import pandas as pd
import datetime
from tweepy import OAuthHandler
access_token = "2885212512-QTrktLlyrLaU5dzGbMjCxU2zAs7SIL8MZDIND9e"
access_token_secret = "Dz86R9UIdQjTkISnfitn1QzsjSN0Xa3eVt9gdmR0Acj5H"
consumer_key = "e5zuECdmrwqJzO4COMh20Dfh8"
consumer_secret = "BtC6qVWwd07YmBuRA1TEfMLinH8hIMuqabezMSQO7LUkfJNUzV"
def collectTwitterData(date):
    today = datetime.date.fromisoformat(date) + datetime.timedelta(days=1)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)
    yesterday= today - datetime.timedelta(days=1)
    tweets_list = tweepy.Cursor(api.search_tweets, q="#india since:" + str(yesterday)+ " until:" + str(today),tweet_mode='extended', lang='en').items(100)
    output=[]
    with open('tweets.txt', 'w',encoding='utf-8') as f:
        for tweet in tweets_list:
        	text = tweet._json["full_text"]
        	favourite_count = tweet.favorite_count
        	retweet_count = tweet.retweet_count
        	created_at = tweet.created_at
        	line = {'text' : text, 'retweet_count' : retweet_count, 'created_at' : created_at}
        	output.append(line)
        	f.write(str(tweet))
        	f.write("\n")
    
    
    df = pd.DataFrame(output)
    df.to_csv('output.csv', mode='a', header=False)

#collectTwitterData('2021-12-02')    