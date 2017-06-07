from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
from nltk import RegexpParser

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)


def process_content():
    try:
        for i in tokenized[:5]:
            tagged = pos_tag(word_tokenize(i)) # tagset='universal'
            chunkGram = r"""Chunk : {<.*>+} 
                        }<VB.?|IN|DT|TO>{"""
            chunkParser = RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            print(chunked)
            for subtree in chunked.subtrees(filter=lambda t:t.label() == "Chunk"):
                print(subtree)
            chunked.draw()                
    except Exception as e:   
        print(str(e))

process_content()        