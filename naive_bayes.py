from sklearn.naive_bayes import MultinomialNB
import tosparse
import numpy

# test data
features = [[(1,2),(2,2)],[(1,0),(2,2)]]
f2 =[[(0,1),(1,9)],[(1,2),(2,1)]]
dict_size = 3
y = [1,2]

# real data
ifile = open('Features_2/BOW vectors/stance_output.txt')
stance_list = [int(line.strip()) for line in ifile]


m1 = tosparse.toSparse(features, 3)
m2 = tosparse.toSparse(f2,3)
print(type(m1))

#nb = MultinomialNB().fit(, y)
#print(nb. predict(tosparse.toSparse([[(1,2),(2,2)]],3)))






