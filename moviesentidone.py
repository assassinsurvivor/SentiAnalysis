import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier 
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier,PassiveAggressiveClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode as m 







class voteclassifier(ClassifierI): 
    def __init__(self,*classifiers):#classifiers hai
        self._classifiers=classifiers

    def classify(self,features):
        votes=[]
        for i in self._classifiers:
            v=i.classify(features)
            votes.append(v)
        return m(votes)
        
    def confidence(self,features):
        votes=[]
        for i in self._classifiers:
            v=i.classify(features)
            votes.append(v)
        
        choice_votes=votes.count(m(votes))
        conf=float(choice_votes)/len(votes)
        return conf



documents= [(list(movie_reviews.words(fileid)),category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]


random.shuffle(documents)






#print(documents[0]) positive document[1] negative
all_words=[]
for i in movie_reviews.words():
     all_words.append(i.lower())
all_words=nltk.FreqDist(all_words)
#print(all_words.most_common(15))
#print(all_words["cheerleader"])#cheerlander nahi tha
word_features=list(all_words.keys())[:4000]
#print (word_features)
def find_feature(document):
    words=set(document)
    features={}
    for i in word_features:
        #print i
        features[i]=(i in words)
    return features

#print ((find_feature(movie_reviews.words("pos/cv010_29198.txt"))))

featuresets=[(find_feature(rev),category)
             for(rev,category) in documents]



#Naive bayes classifier
#likelihood=(prior occurence*likelihood)/evidence

training_set=featuresets[:1950]
testing_set=featuresets[1950:]
classifier=nltk.NaiveBayesClassifier.train(training_set)


print("Naive bayes classifier accuracy in percentage:",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(11)

save_classifier=open("c:/users/gaurav/desktop/naiveclassifier.pickle","w")
pickle.dump(classifier,save_classifier)
save_classifier.close()


MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("original Mnb classifier accuracy in percentage:",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)

save_classifier=open("c:/users/gaurav/desktop/mnbclassifier.pickle","w")
pickle.dump(MNB_classifier,save_classifier)
save_classifier.close()

Bern_classifier=SklearnClassifier(BernoulliNB())
Bern_classifier.train(training_set)
print("bernoulli classifier accuracy in percentage:",(nltk.classify.accuracy(Bern_classifier,testing_set))*100)

save_classifier=open("c:/users/gaurav/desktop/BernoulliNBclassifier.pickle","w")
pickle.dump(Bern_classifier,save_classifier)
save_classifier.close()

logistics_classifier=SklearnClassifier(LogisticRegression())
logistics_classifier.train(training_set)
print("logistics classifier accuracy in percentage:",(nltk.classify.accuracy(logistics_classifier,testing_set))*100)


save_classifier=open("c:/users/gaurav/desktop/LogisticRegressionclassifier.pickle","w")
pickle.dump(logistics_classifier,save_classifier)
save_classifier.close()


SGDClassifier_classifier=SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
SGDClassifier_classifier aayega
print("original SGDC classifier accuracy in percentage:",(nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)

save_classifier=open("c:/users/gaurav/desktop/SGDClassifier.pickle","w")
pickle.dump(SGDClassifier_classifier,save_classifier)
save_classifier.close()

PassiveAggressiveClassifier_classifier=SklearnClassifier(PassiveAggressiveClassifier())
PassiveAggressiveClassifier_classifier.train(training_set)
print("original passive Aggressve classifier accuracy in percentage:",(nltk.classify.accuracy(PassiveAggressiveClassifier_classifier,testing_set))*100)

save_classifier=open("c:/users/gaurav/desktop/PassiveAggressiveClassifier.pickle","w")
pickle.dump(PassiveAggressiveClassifier_classifier,save_classifier)
save_classifier.close()


SVC_classifier= SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("original passive Aggressve classifier accuracy in percentage:",(nltk.classify.accuracy(SVC_classifier,testing_set))*100)

save_classifier=open("c:/users/gaurav/desktop/SVCclassifier.pickle","w")
pickle.dump(SVC_classifier,save_classifier)
save_classifier.close()

voted_classifier = voteclassifier(classifier,
                            MNB_classifier,
                            Bern_classifier,
                            logistics_classifier,
                            SGDClassifier_classifier,
                            SVC_classifier,
                            PassiveAggressiveClassifier_classifier
                            )


print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)


def sentiment(text):
    feats = find_feature(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
