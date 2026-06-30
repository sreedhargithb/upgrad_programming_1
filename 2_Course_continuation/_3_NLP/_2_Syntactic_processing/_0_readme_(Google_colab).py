#!/usr/bin/env python
# coding: utf-8

# ## _1_Intro_to_Syntactic_processing_and_POS_tagging

# - Run in Google colab (https://colab.research.google.com/drive/1Y59MnjxU65X6Zum9_65apJLtlV_47avn#scrollTo=tKh2uQj5etaw)
# 
# <code>
# import spacy
# nlp = spacy.load("en_core_web_sm")
# doc = nlp("upGrad is teaching Data Science courses to the working professionals.")
# for token in doc:
#     print(token.text, token.pos_, token.tag_)
# </code>
# 
# - You can take a look at the universal tagsets used by the spaCy toolkit here (https://universaldependencies.org/docs/u/pos/).
# 
# - spaCy (https://spacy.io/) is an open-source library used for advanced natural language processing, similar to NLTK, which you have used in lexical processing.
# 
# - You can also refer to the alphabetical list of 36 part-of-speech tags used in the Penn Treebank Project (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html), which is being used by the spaCy library.

# <pre>
# You have learnt the following two types of PoS taggers:
# 
# Rule-based tagger: Here, you assign the most frequent PoS tags that appear in the training data to the test data set. However, sometimes, it does not give satisfactory results because it does not incorporate the context of the word.
# Sequential tagger (Hidden Markov Model): Sequence labelling is the task of assigning respective PoS tags of words in a sentence using the PoS tag of the previous word in that sentence.
# </pre>

# ## _2_Parsing

# <hr>
# <pre>
#     - <a href="https://explosion.ai/demos/displacy?text=Economic%20news%20had%20little%20effect%20on%20financial%20markets&model=en_core_web_sm&cpu=0&cph=0">displaCy Dependency Visualizer:</a>
#     - <a href="https://hemingwayapp.com/">The Hemingway application identifies the various instances where the document needs to be corrected grammatically. </a>
# </pre>

# In[ ]:




