import spacy
from spacy import displacy
from spacy.matcher import Matcher
import pandas as pd
nlp = spacy.load("en_core_web_sm")

active_passive = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_3_NLP/_2_Syntactic_processing/_2_Parsing/_1_Dependency_Parsing_(Google_colab)/Dataset/active_passive.csv')
active_passive.head(2)

active_passive.shape

active = active_passive['Active']
passive = active_passive['Passive']

passive_rule = [{'DEP':'nsubjpass'}]
matcher = Matcher(nlp.vocab)
matcher.add('Rule',[passive_rule])

def is_passive(doc,matcher):
    if len(matcher(doc))>0:
        return True
    else:
        return False

cnt = 0
for sent in active:
    doc = nlp(sent)
    if not is_passive(doc,matcher):
        cnt += 1
print(cnt)

cnt = 0
for sent in passive:
    doc = nlp(sent)
    if is_passive(doc,matcher):
        cnt += 1
print(cnt)

cnt = 0
missed = []
for sent in passive:
    doc = nlp(sent)
    if is_passive(doc,matcher):
        cnt += 1
    else:
        missed.append(doc)
print(cnt)

missed[0]

missed[1]

for doc in missed:
    displacy.render(doc, style="dep")

spacy.explain("auxpass")

passive_rule = [{'DEP':{"IN":['nsubjpass','auxpass']}}]
matcher = Matcher(nlp.vocab)
matcher.add('Rule',[passive_rule])

cnt = 0
for sent in active:
    doc = nlp(sent)
    if not is_passive(doc,matcher):
        cnt += 1
print(cnt)

cnt = 0
missed = []
for sent in passive:
    doc = nlp(sent)
    if is_passive(doc,matcher):
        cnt += 1
    else:
        missed.append(doc)
print(cnt)

import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))
