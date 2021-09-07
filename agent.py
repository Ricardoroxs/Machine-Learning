import numpy as np
from sklearn.naive_bayes import MultinomialNB

#-------
# dictionary, to look up words from the data vector -- case sensitive! 
#-------
dictionary = np.array(["congrats","you","are","selected","won","lottery","travel","for","free","credit","cards","very","good","night"])

#-------
# vec2word: convert data vector to words
#-------
def vec2word(vec):
  """
  arguments: vec = np.array([0,1,...])
  returns: string of sentence corresponsing to the vector (word may not be ordered properly)
  """
  str = ""
  index = 0
  for s in dictionary:
    if vec[0][index] == 1:
      str = str + dictionary[index] + " "
    index += 1
  #print(str)
  return str


#--------------------------------
# spam data : enter your data here (SOL)
#--------------------------------
X = np.array([
 [1,1,1,1,0,0,0,0,0,0,0,0,0,0],
 [1,1,0,0,1,1,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,1,1,1,0,0,0,0,0],
 [0,0,0,1,0,0,0,1,0,1,1,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,1,1,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,1,1],
 [0,0,0,0,0,1,0,0,0,0,0,0,0,0]
])

y = np.array([0,1,1,1,0,0,1])

clf = MultinomialNB()
clf.fit(X, y)

print("Score (accuracy: 1.0 = 100%)= ",end="")
print(clf.score(X,y))

print("Spam test: you won lottery: answer=",end="")
test = np.array([[0,0,0,1,0,0,1,0,0,0,0,0,1,1]]) # note: np.array([[ ... ]]), not np.array([ ... ]) 
# test your word2vec() function here, with the test data. 
print(vec2word(test))
print(clf.predict(test))
print(clf.predict_proba(test))  # [1. 0.] correspond to NoSpam and Spam probability, respectively.
