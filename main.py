import tweepy
import requests
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import random
import time
from datetime import date
from datetime import timedelta
from collections import Counter
import numpy as np
import math
from newspaper import Article
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def get():

    out = ""
  
    domains = ",".join(random.sample(["bbc.com", "bloomberg.com", "cbsnews.com", "cnn.com", "espn.com", "foxnews.com", "msnbc.com", "nbcnews.com", "politico.com", "reuters.com", "wsj.com", "washingtonpost.com", "time.com","usatoday.com"], 3))
    url = ("https://newsapi.org/v2/everything?"
           f"domains={domains}&"
           f"from={(date.today() - timedelta(days = 1)).strftime('%Y-%m-%d')}&"
           "language=en&"
           "sortBy=publishedAt&"
           "apiKey=b2bb907034144e90a2a356c1f9b88bdf")
    response = requests.get(url)
    
    good = False
    articles = response.json()["articles"]
    
    n = 0
    
    while not good and n < len(articles):
      article = articles[n]
      description = article["description"]
      a = Article(article["url"])
      a.download()
      a.parse()
      hashtags = get_hashtags(a.text.replace("\n", " "))
      if len(hashtags) == 0: #Ensure output has one or more hashtags
        n += 1
        continue
      tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
      description = " ".join(tokenizer.tokenize(description)[:-1]) #last sententence always ends in ellipsis, so remove it
      title = article["title"]
      url = article["url"]

      questions = set()
      for keyword in ["top", "#1", "#2", "#3", "most relevant"]: #Rephrase question to model to get varied outputs
        for txt in [description, title]: #Rephrase question to model with both description and title to get varied outputs
          response = requests.post(
              "https://api.ai21.com/studio/v1/j1-large/complete",
              headers={"Authorization": "Bearer L8lXNx2u7PHmvuVZNTNjy4ZBy9egXNwB"},
              json={
                  "prompt": f"What is your {keyword} question after learning that {txt}. My {keyword} question is:", 
                  "stopSequences": ["?"],
                  "topKReturn": 0,
                  "temperature": 0.0
              }
          )
          q = ""
          res = response.json()["completions"][0]["data"]["text"]
          for char in res:
            if char not in ["#", "<"]: #Model often outputs these symbols due to HTML in description or attempt to repeat numeric keywords. Ignore these cases.
              if len(q) == 0 and char.isalpha(): #Make sure first letter in question is capitalized
                q += char.upper()
              elif len(q) > 0:
                q += char
            else:
              q = ""
              break
          if "question" not in q and "cost" not in q and "calling" not in q and len(q.split()) > 4: #Don't include output rephrasing input or several common responses. Additionally, ensure question is at least five words.
              questions.add(q)
      questions = list(questions)

      i = 0
      while i < len(questions) - 1: #Iterate through list to check for similar questions
          q1 = set(questions[i].split())
          j = i + 1
          while j < len(questions): #Iterate through other words starting at next word
              q2 = set(questions[j].split())
              if (2 * len(q1 & q2) / (len(q1) + len(q2))) > 0.5: #If over 50% of words overlap, remove the shorter question
                  if len(q1) < len(q2):
                      questions.pop(i)
                      break
                  else:
                      questions.pop(j)
                      j -= 1
              j += 1
          i += 1
      if len(questions) > 0: #Set output and break out of loop if there is 1 or more question remaining
          good = True
          random.shuffle(questions) #Randomize questions
          questions = questions[:min(random.randint(1, 3), len(questions))] #Pick up to three questions
          if len(questions) == 1: #Format as appropriate
              out += f"{questions[0]}?\n"
          else:
              random.shuffle(questions)
              for i in range(len(questions)):
                  out += f"{i+1}. {questions[i]}?\n"
          if len(hashtags) > 0:
            out += "\n" + " ".join(hashtags) #Add hashtags
          out += f"\n{url}"
          break
      n += 1
    return out

def get_hashtags(input):
    hashtags = Counter()
    nltk_results = ne_chunk(pos_tag(word_tokenize(format(input)))) #Chunk words
    for nltk_result in nltk_results:
        if type(nltk_result) == Tree or nltk_result[1] == "NNP": #Add hashtag for proper nouns
            if type(nltk_result) == Tree:
              ht = "#"
              for leaf in nltk_result.leaves():
                if leaf[1] == "NNP":
                  cap = False
                  for i in range(len(leaf[0])):
                    if not leaf[0][i].isalpha():
                      cap = True
                    elif cap == True and leaf[0][i].isalpha():
                      ht += leaf[0][i].upper()
                      cap = False
                    else:
                      ht += leaf[0][i]
              if len(ht) > 2 and ht not in ["#Mr", "#Mrs", "#Ms"]:
                hashtags.update([ht])
            elif type(nltk_result) != Tree:
              ht = "#"
              cap = False
              for i in range(len(nltk_result[0])):
                if not nltk_result[0][i].isalpha():
                  ht = "#"
                  break
                else:
                  ht += nltk_result[0][i]
              if len(ht) > 2 and ht not in ["#Mr", "#Mrs", "#Ms"]:
                hashtags.update([ht])
    keys = list(hashtags.keys())
    i = 0
    while i < len(keys): #Use longest version of names for hashtags
      k1 = keys[i]
      for k2 in (keys[:i] + keys[i+1:]):
        if k1[1:] in k2[1:]:
          hashtags.update({k2: hashtags[k1]})
          del hashtags[k1]
          keys.remove(k1)
          break
        elif k2[1:] in k1[1:]:
          hashtags.update({k1: hashtags[k2]})
          del hashtags[k2]
          keys.remove(k2)
      i += 1
    cutoff = max(3, np.percentile(list(hashtags.values()), 90)) #Only add hashtags for words that appear at least 3 times and are in at least the 90th percentile of appearances for all proper nouns
    return [i[0] for i in sorted([(k, v) for k, v in hashtags.items() if v >= cutoff], key = lambda i: i[1], reverse = True)]

def format(text):
  punctuation = [".", "!", "?"]
  titles = ["Mr.", "Ms.", "Mrs.", "Dr."]
  sentences = []
  curr = []
  for word in text.split(" "): #Loop through words in text
    if len(word) != 0:
      word  = word.lstrip()
      if word not in titles: #Ignore titles such as Mr., Mrs., etc.
        curr.append(word) #Add word to current sentence
        if word[-1] in punctuation: #Check if word ends in punctuation
          sentences.append(" ".join(curr)) #Add sentence to list of sentences
          curr = [] #Reset current sentence
  for i in range(len(sentences)):
    first = ne_chunk(pos_tag(word_tokenize(sentences[i])))[0]
    if not(type(first) == Tree and first.label() in ["PERSON", "ORGANIZATION"]): #Change first letter to lowercase unless proper noun
      sentences[i] = sentences[i][0].lower() + sentences[i][1:]
  return " ".join(sentences) #Formatted text

CONSUMER_KEY = "[REDACTED]"
CONSUMER_SECRET = "[REDACTED]"
ACCESS_KEY = "[REDACTED]"
ACCESS_SECRET = "[REDACTED]"

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
api = tweepy.API(auth)

while True:
  try:
    content = get()
    api.update_status(content) #Send out a Tweet
    print(content, end = "\n\n")
  except:
    pass
  time.sleep(900) #Wait 15 minutes in betweeen calls to News API to avoid maxing out daily limit
