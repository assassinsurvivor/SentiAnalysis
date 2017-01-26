from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import temp as t

ckey="7LiieRK4V9OnPEpNr6TS9wN5v"
csecret="ZAAYDBiYZZf6BBFTg9zfnFE9PBzmFj9Fw4ySGzAq0kvdM2sfLi"
atoken="818098577616596992-bGuHdprMlVFx2n0QYYB70PTsnPD165y"
asecret="cAjc4Rt0qQ9WkBDtJRomn6Bw07xGGXbjZ5tW9nkgbM34E"

class listener(StreamListener):

    def on_data(self,data):
        all_data=json.loads(data)
        tweet=all_data["text"]
        sentiment_value,confidence=t.sentiment(tweet)
        print((tweet,sentiment_value,confidence))
        
        
        if confidence*100 >=80:
            output=open("c:/users/ashub/desktop/twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()
        return True
    def on_error(self,status):
        print(status)

auth=OAuthHandler(ckey,csecret)
auth.set_access_token(atoken,asecret)

twitterstream=Stream(auth,listener())
twitterstream.filter(track=["tensorflow"])
