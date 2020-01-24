# Documentation

## Overview

"Just Ask Jane" is a retrieval-based chatbot fed with the complete works of Jane Austen. 

Primarily for enterainment, this bot could also be useful to students of literature for finding passages 
in Austen's work concerning topic words entered by the user.

It could also be used as a base for further literary explorations by, for example, training it to return 
all passages concerning the topic input.

## Usage

1. Open the following URL in your browser: https://join.slack.com/t/justaskjane/signup

2. You will be prompted to log in. You must log in with a illinois.edu domain in order to join the channel!

3. Once logged on, click on "Direct Messages" on the left-hand side.

4. In the input "Find or start a conversation", type "Jane". Select the result, and click "go".

5. You can now chat with Jane. Type @Jane and then your request. For example:

"Tell me about happiness"

"What makes for good health?"

"What do you like to eat?"

Jane sometimes takes a few seconds to answer!

## Implementation

The implementation consists of the following parts: 

1. Preparing the text for modelling
2. Training the model: parameter optimization and measuring similarity
4. Request-response mechanism for user to interact with the bot
5. Provisioning of this mechanism over Slack, served with the help of components hosted on AWS

### Preparing the text for modelling

Code for text preparation for modelling is found in **train/jane-simple.ipynb**. It consists of a function called **clean_text()**, which
takes a string and transforms it to lowercase, removes special characters, book title, author, volume and chapter headings.
I don't remove punctuation in this step because I still want to keep sentences as text entities at this point, and I will need 
specific punctuation marks for later to discern dialog.

We loop through all the books we've saved as text files in the data directory, and build one long text string from the cleaned
contents of those text files by passing each text into the **clean_text()** function.

Now the text is ready to be tokenized. I do this in two steps: the first is to use NLTK, a Python natural language processing 
library, to tokenize the one long string into a list of sentences. The second step is to extract the dialogue from these sentences,
using a function I created called **extract_dialog()**. This function accepts a tokenized sentence, searches it for the 
punctuation markers that indicate dialogue, extracts the text between these markers (if the sentence is a dialogue) and returns it.
I call this function in order to make a list of dialogue from our NLTK tokenized sentences. Text that is not dialogue is discarded.

I ended up doing it this way after a lot of experimentation with different parameters on models of the complete text. With the 
complete text, interaction with the bot resulted in responses that were correct, but not conversational. I tried different things
like mapping pronouns from the user request to the opposite pronouns for the bot response (for example, "what was your mother like"
would be mapped to "my mother...was like") in order to return something that FELT like a response instead of an echo. I came
to the conclusion that for "natural" sounding dialog, it would make sense to only pull text that was already in dialogue format.
The answers continued to be fairly precise, but gave the interaction a more "conversational" feeling than when I had been using
complete text for the model.

Additionally, using only Austen's dialogue instead of the complete text reduced the size of the finished model, 
which was helpful in keeping the bot response time down. This was also critical for hosting the model on AWS Lambda!

### Training the model:

I used Python's sklearn and pandas libraries to create a TFIDF vector model of the dialogue sentences, and a dataframe 
to store the weights. After much experimenting with parameters, I found that using the default English stopwords for the
TFIDF vectorizer worked the same as for any subset of stopwords, but reduced the size of the finished model the most.
Additionally, I found that creating my own lemmatizer did not end up with significantly different results as the 
default tokenizer used by sklearn's TFIDF vectorizer. Using the default instead of my own lemmatizer additionally
removed an obstacle to hosting the finished model on AWS Lambda, which would have required making the custom lemmatizer
a portable Python module and extra provisioning in order to host it. 

The pandas dataframe is a representation of our TFIDF weights: each column is a unique
word from our tokens, and each row is "document", in our case, a sentence of dialogue. The dataframe is human-readable
and I use it when measuring cosine similarity of the user request to our dialogues. This is implemented in the **response()**
function: first the user request is transformed using the TFIDF model, then the dot product of this transformation 
is calculated on the transpose of the pandas dataframe. Calling Python's argmax() function on this result (the cosine similarity)
results in an index that we can then plug into our dialogue matrix to return the most matching sentence to the user request. 

This solution was much preferable to what I had otherwise seen in most tutorials elsewhere,
where the sentance token list would be appended to with the user request and the TFIDF model then REFIT to that new data.
This method works, but I don't want to fit the model every time the user makes a request, because it would be so
incredibly slow! I took a solution that requires fitting the model ONCE, using the model after the initial fitting
 only to transform the user query, and then calculating the cosine similarity on a static dataframe.


### Interacting with the bot: request-response mechanism

For local testing I used a request-response mechanism I had found in a tutorial: this simply accepts user input, plugs it into
the **response()** function, and returns the result of that function call. This is repeated until a determined point (in
my case, "Thank you" from the user). I only used this mechanism locally to see the results of my model. The mechanism that
actually serves the request-response for interaction with the bot can be found in **architecture/lambda_function.py** and
replaces the simple flag and user input() setup with other components from AWS.

### Provisioning

Getting the chatbot to work on Slack was harder and more time-consuming that the rest of the steps already described.
You need to setup a Slack workspace and create the initial chatbot app, which is the easy part. Then on AWS, you need
to set up an endpoint over Gateway API to receive POST requests from slack. The Lambda component hosts the code
for fetching the user request from the endpoint, verifying it, and then serving the response back to Slack.

#### Obstacles:

There were multiple time-consuming, frustrating obstacles, including setting up layers in Lambda with the 
Python sklearn and pandas libraries, which needed to be in a format specific to the Lambda runtime environment 
(see **architecture/lambda_layers** and **build** directory files) and not too large for Lambda to handle.

The models I trained in 
**train/jane-simple.ipynb** needed to be serialized, uploaded to S3, fetched in **lambda_function.py**, and 
deserialized before they could be used in the code for transforming the request and serving the response. 
(Pickled models can be found in the **architecture/lambda_models** directory). But Lambda had an issue with the size of 
the initial models using complete text, and an issue with the recommended serialization library, joblib, and initial
problems with permissions required to access other AWS components. 

Finally, after creating a model I was satisfied with that Lambda could deal with, size-wise, 
there was a major issue with the Lambda function invoking multiple
times, which resulted in the chatbot on the Slack side responding repeatedly to one request. This required additional
tooling using a DynamoDB component to persist the request ID and check it before serving a request.
