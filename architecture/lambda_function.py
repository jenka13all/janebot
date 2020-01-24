#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jennifer
"""

import os
import urllib
import urllib.request
import json
import boto3
import pickle
import re

#get environment vars for bot_token and verification_token
BOT_TOKEN = os.environ["BOT_TOKEN"]
VER_TOKEN = os.environ["VERIFICATION_TOKEN"]

#define api endpoint to send our responses to
SLACK_URL = "https://slack.com/api/chat.postMessage"

#fetch our trained model components from s3
s3 = boto3.client('s3', region_name='us-east-1')

s3_response = s3.get_object(Bucket="jane-chatbot-models", Key="model_simple_tfidf.pkl")
tfidf = pickle.loads(s3_response['Body'].read())

s3_response = s3.get_object(Bucket="jane-chatbot-models", Key="model_simple_dialog.pkl")
dialog = pickle.loads(s3_response['Body'].read())

s3_response = s3.get_object(Bucket="jane-chatbot-models", Key="model_simple_X.pkl")
X = pickle.loads(s3_response['Body'].read())

#set up connection to dynamo db
dynamodb = boto3.client('dynamodb')

def lambda_handler(event, context):
    #verify that post is coming from slack
    if event["token"] != VER_TOKEN:
        print("Token not verified!")
        
        #return HTTP unauthorized response
        return {
            'statusCode': 401,
            'body': json.dumps('Unauthorized')
        }
        
    #only need on Slack App creation    
    #if "challenge" in data:
    #    return event["challenge"]
        
    #get the slack event data    
    slack_event = event['event']

    #ignore the event if it comes from the bot
    if "bot_id" in slack_event:
        print("Ignore bot event")
    else:
        #get the client_msg_id of the incoming request
        #lambda has an issue where the function is invoked multiple times, sending up to 3 responses to the bot
        #if we find the client_msg_id already in our dynamo db, we can keep the function from being re-invoked
        client_msg_id = slack_event['client_msg_id']
        
        #stop here if we've already been through this for this client_msg_id
        already_posted = dynamodb.get_item(TableName='ClientMessages', Key={'client_msg_id':{'S':client_msg_id}})

        if 'Item' in already_posted and len(already_posted['Item']) > 0:
            print("Ignore identical client_msg_id ", client_msg_id)
        else:
            #insert the new client_msg_id into the database
            dynamodb.put_item(TableName='ClientMessages', Item={'client_msg_id':{'S':client_msg_id}})

            #get the user request text: strip the non-request stuff from it first and transform to lowercase
            user_id = r"<@\w+>"
            text = re.sub(user_id, "", slack_event['text']).strip()
            text = text.lower()

            #hard-coded greeting responses
            if text in ['hello', 'hi']:
                response = 'Good day!'
            elif re.search('how are you', text) is not None:    
                response = 'Exceedingly well, thank you!'
            else:
                #ignore 'tell me about...' part of any request: we don't want that to be measured for similarity
                text = text.replace('tell me about', '')
                
                #find highest similarity of user query to our model
                query = tfidf.transform([text])
                cosine_sim = query.dot(X.T)
        
                #if no similarity found, give a standard response
                if cosine_sim.argmax() == 0:
                    response = 'I beg your pardon? I\'m not quite sure I got your meaning.'
                else:
                    #return the most similar match from our dialog matrix
                    response = dialog[cosine_sim.argmax()]
                    response = response[0:-1] + '.'
                    response = response.capitalize()

            channel_id = slack_event["channel"]
        
            #response back to Slack needs the text, channel_id to send it to, and OA Auth token
            #in a form that Slack can handle (not json)
            data = urllib.parse.urlencode(
                (
                    ("token", BOT_TOKEN),
                    ("channel", channel_id),
                    ("text", response)
                )
            )
            data = data.encode("ascii")

            #tell slack not to post multiple times!
            header = { 
                'Content-Type' : 'application/x-www-form-urlencoded"', 
                'X-Slack-No-Retry' : 1 
            }
        
            #add all this info to the request and send it off
            request = urllib.request.Request(
                SLACK_URL, 
                data=data, 
                method="POST",
                headers=header
            )
        
            urllib.request.urlopen(request).read()

    #for when everything went find
    return {
        'statusCode': 200,
        'body': json.dumps(''),
        'headers': {'X-Slack-No-Retry': 1}
    }
