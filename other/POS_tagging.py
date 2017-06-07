from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:5]:
            #words = word_tokenize(i)
            #tagged = pos_tag(words)
            print(pos_tag(word_tokenize(i))) # tagset='universal'
    except Exception as e:   
        print(str(e))

process_content()        