import relevancer as rlv
import pandas as pd
import time
from sklearn.naive_bayes import MultinomialNB
from datetime import date

collection = 'healthtags_id_clusters'
df = pd.DataFrame(columns=['label', 'text'])
index = 0
my_token_pattern = r"[#@]?\w+\b|[\U00010000-\U0010ffff]"
today = date.today()
pickle_file = today.strftime('%Y%m%d') + '_' + collection + '_classifier'

rlvdb, rlvcl = rlv.connect_mongodb(configfile='data/mongodb.ini',coll_name=collection) #Db and the collection that contains the tweet

for cluster in rlvcl.find({'classes': {'$ne': None}}):
   label = cluster['classes']['source']
   index += 1
   print(index)
   for tag in cluster['ctweettuplelist']:
      # print(tag[1])
      df.loc[df.size//2] = [label, tag[2]]

vect_and_classifier = rlv.get_vectorizer_and_mnb_classifier(df, my_token_pattern, pickle_file=pickle_file)

print(df);
# print(df2);