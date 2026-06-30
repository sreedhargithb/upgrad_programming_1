import spacy # import spacy module

model = spacy.load("en_core_web_sm") #load pre-trained model

doc = "Sumit is an adjunct faculty at Upgrad. "

processed_doc = model(doc); #process input and perform NLP tasks






doc2 = "Dr. Sumit is an adjunct faculty at UpGrad"
processed_doc2 = model(doc2)

for token in processed_doc2:
  print(token.text, " -- ", token.pos_)

# Print the entities in processed_doc2


doc3 = "Statue of Liberty is situated in New York, USA."
processed_doc3 = model(doc3)

for token in processed_doc3:
  print(token.text, " -- ", token.pos_)

# Print the entities in processed_doc3


# Print the IOB format.


email = ('Dear Family, Jose Luis and I have changed our dates, we are '
         'going to come to Aspen on the 23rd of December and leave on the '
         '30th of December. We would like to stay in the front bedroom of '
         'the Aspen Cottage so that Mark, Natalie and Zachary can stay in '
         'the guest cottage. Please let me know if there are any problems '
         'with this. If I do not hear anything, I will assume this is all '
         'o.k. with you.'
         'Love, Liz')

processed_email = model(email) # Apply spacy's model to process the email

# intialize data structure to store anonymized email
# Write your code here.

for ent in processed_email.ents:
  if(ent.label_ == 'PERSON'): # if the word corresponds to a PERSON entity
    for char_pos in range(ent.start_char, ent.end_char): # use character positions
      anonymized_email[char_pos] = '*'

# Print the email after anonymisation



