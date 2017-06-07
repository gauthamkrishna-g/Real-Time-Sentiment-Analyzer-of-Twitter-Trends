from nltk.corpus import  wordnet

syns = wordnet.synsets("program")
print(syns[0].name())
print(syns[0].lemmas()[0].name())
print(syns[0].definition())
print(syns[0].examples())   

synonyms = []
antonyms = []

for syn in wordnet.synsets("cool"):
    for l in syn.lemmas():
        #print("l:", l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('truck.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('hello.n.01')
w2 = wordnet.synset('phone.n.01')
print(w1.wup_similarity(w2))