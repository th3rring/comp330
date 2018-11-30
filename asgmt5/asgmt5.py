import re
import math
import numpy as np
from collections import defaultdict

def create_word_dict(url):
    corpus = sc.textFile(url)
    validLines = corpus.filter(lambda x : 'id' in x)
    keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))
    regex = re.compile('[^a-zA-Z]')
    keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))
    allCounts = allWords.reduceByKey(lambda a, b: a + b)
    topWordsUnsorted = allCounts.top(20000, lambda x: x[1])
    topWords = sorted(topWordsUnsorted, key=lambda x: (-x[1], x[0]))
    twentyK = sc.parallelize(range(20000))
    word_dict = twentyK.map (lambda x: (topWords[x][0], x))
    return word_dict.cache().sortByKey()

# word_dict = create_word_dict("s3://chrisjermainebucket/comp330_A5/SmallTrainingDataOneLinePerDoc.txt")
word_dict = create_word_dict("s3://chrisjermainebucket/comp330_A5/TrainingDataOneLinePerDoc.txt")

#Prints 347
word_dict.lookup("applicant")[0]

#Prints 2
word_dict.lookup("and")[0]

#Prints 504
word_dict.lookup("attack")[0]

#Prints 3014
word_dict.lookup("protein")[0]

#Prints 612
word_dict.lookup("car")[0]

# Convert the current python array into a numpy data structure
def npArrayConvert(list1):
    array = np.zeros(20000)
    for thing in list1:
        array[thing] += 1
    return array


# Sets elements to 1 if word is in document
def inDocument(vector):
    for i in range(20000):
        if vector[i] > 1:
            vector[i] = 1
    return vector


def vectorMult(tf,idf):
    tf_idf1 = np.zeros(20000)
    for num in range(20000):
        tf_idf1[num] = tf[num]*idf[num]
    return tf_idf1

def create_tf_idf_normalized(url,idf):
    corpus = sc.textFile(url)
    validLines = corpus.filter(lambda x : 'id' in x)
    keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))
    regex = re.compile('[^a-zA-Z]')
    keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))
    allCounts = allWords.reduceByKey(lambda a, b: a + b)
    topWordsUnsorted = allCounts.top(20000, lambda x: x[1])
    topWords = sorted(topWordsUnsorted, key=lambda x: (-x[1], x[0]))
    twentyK = sc.parallelize(range(20000))
    word_dict = twentyK.map (lambda x: (topWords[x][0], x))
    word_docs = keyAndListOfWords.flatMap(lambda x: ((word, str(x[0])) for word in x[1]))
    temp = word_dict.join(word_docs)
    temp.groupByKey().map(lambda x: (x[0], list(x[1])))
    doc_ranks = temp.map(lambda x: (x[1][1], x[1][0]))
    rdd_map = doc_ranks.groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda x: (x[0], npArrayConvert(x[1])))
    tf = rdd_map.map(lambda x: (x[0], x[1]/np.sum(x[1])))
    tf_idf = tf.map(lambda x: (x[0], vectorMult(x[1], idf)))
    means = tf_idf.values().sum() / tf_idf.count()
    # return tf.map(lambda x: (x[0], vectorMult(x[1], idf)))
    std_dev = np.sqrt(tf_idf.map(lambda x: np.square(x[1]-means)).reduce(lambda a, b:a+b) / float(tf_idf.count()))
    return  tf_idf.map(lambda x: (x[0], np.nan_to_num((x[1]-means)/std_dev))).cache().sortByKey()

def create_idf(url):
    corpus = sc.textFile(url)
    validLines = corpus.filter(lambda x : 'id' in x)
    keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))
    regex = re.compile('[^a-zA-Z]')
    keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))
    allCounts = allWords.reduceByKey(lambda a, b: a + b)
    topWordsUnsorted = allCounts.top(20000, lambda x: x[1])
    topWords = sorted(topWordsUnsorted, key=lambda x: (-x[1], x[0]))
    twentyK = sc.parallelize(range(20000))
    word_dict = twentyK.map (lambda x: (topWords[x][0], x))
    word_docs = keyAndListOfWords.flatMap(lambda x: ((word, str(x[0])) for word in x[1]))
    temp = word_dict.join(word_docs)
    temp.groupByKey().map(lambda x: (x[0], list(x[1])))
    doc_ranks = temp.map(lambda x: (x[1][1], x[1][0]))
    rdd_map = doc_ranks.groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda x: (x[0], npArrayConvert(x[1])))
    tf = rdd_map.map(lambda x: (x[0], x[1]/np.sum(x[1])))
    num_docs = keyAndListOfWords.count()
    word_in_doc = rdd_map.map(lambda x: (1, inDocument(x[1])))
    # Finds total bumber of docs in which words appear in and then reduces by key
    all_words_in_doc = word_in_doc.aggregateByKey(np.zeros(20000), lambda a, b: a+b, lambda a,b: a+b)
    denominator = all_words_in_doc.top(1)[0][1]
    return np.log(num_docs/denominator)


def normalize_row(x):
    mean = np.mean(x[1])
    std = np.std(x[1])
    std = 1 if std == 0.0 else std
    return (x[1]-mean)/std
# idf = create_idf("s3://chrisjermainebucket/comp330_A5/SmallTrainingDataOneLinePerDoc.txt")
# tf_idf = create_tf_idf_normalized("s3://chrisjermainebucket/comp330_A5/SmallTrainingDataOneLinePerDoc.txt",idf)
idf = create_idf("s3://chrisjermainebucket/comp330_A5/TrainingDataOneLinePerDoc.txt")
tf_idf = create_tf_idf_normalized("s3://chrisjermainebucket/comp330_A5/TrainingDataOneLinePerDoc.txt",idf)

def grad_row(x,r,reg):
    # theta_i = vectorMult(r, x[1]).sum()
    theta_i = r.dot(x[1])
    y_i = 1 if "AU" in str(x[0]) else 0
    grad = np.vectorize(lambda a: -a*y_i + a*(np.exp(theta_i)/(1+np.exp(theta_i))))
    #Vectorize and make the regularization a vector add
    return grad(x[1]) + 2*reg*r 


def llh_calc(x,r):
    # theta_i = vectorMult(r, x[1]).sum()
    theta_i = r.dot(x[1])
    y_i = 1 if "AU" in str(x[0]) else 0
    return -y_i*theta_i+np.log(1+np.exp(theta_i))

def grad_desc(initial_r, reg, tf_idf):
    #Change to initial guess
    #Reg coef to 0 or close to 0
    r = initial_r 
    #Just sone random initialization for delta loss
    delta = 10000000
    lr = 0.1
    num_docs = tf_idf.count()
    old_llh = (tf_idf.map(lambda x: llh_calc(x,r)).reduce(lambda a, b: a + b) + reg*r.dot(r))/num_docs
    print(old_llh)
    while delta > 0.00001:
        gradient = tf_idf.map(lambda x: grad_row(x,r,reg)).reduce(lambda a, b: a + b)/num_docs
        print(gradient)
        r -= lr*gradient
        new_llh = tf_idf.map(lambda x: llh_calc(x,r)).reduce(lambda a, b: a + b) + reg*r.dot(r)/num_docs
        print(new_llh)
        delta = abs(new_llh-old_llh)
        if(new_llh > old_llh):
            lr *= 1.1
        else:
            lr /= 2
        old_llh = new_llh
    return r

tf_idf_sample = tf_idf.sample(True,0.3)
r_opt = grad_desc(np.zeros(20000),0,tf_idf_sample)
r_opt = grad_desc(r_opt,0,tf_idf)



top_50 = r_opt.argsort()[-50:][::-1] 
word_dict_rev = word_dict.map(lambda x: (x[1],x[0])).cache().sortByKey()
for inx in top_50:
    print(word_dict_rev.lookup(inx))
    

tf_idf_pred = create_tf_idf_normalized("s3://chrisjermainebucket/comp330_A5/TestingDataOneLinePerDoc.txt",idf)

def predict(cutoff):
    y_pred_label = tf_idf.sortByKey().map(lambda x: (x[0], [1] if r_opt.dot(x[1]) > cutoff else [0]))
    y_label = tf_idf.sortByKey().map(lambda x : (x[0], [1] if "AU" in str(x[0]) else [0]))
    y_pred = y_pred_label.values().reduce(lambda a, b: np.append(a,b))
    y = y_label.values().reduce(lambda a, b: np.append(a,b))
    # diff_pred = y_pred_label.join(y_pred).filter(lambda x: x[1][0] == 1 and x[1][1] == 0).keys()
    recall = y_pred.sum()/float(y.sum())
    precision = y_pred.dot(y)/float(y_pred.sum())
    added = y+y_pred
    return (2*precision*recall)/(precision+recall)

predict(0.6)

cutoff = 0.6
y_pred_label = tf_idf.sortByKey().map(lambda x: (x[0], [1] if r_opt.dot(x[1]) > cutoff else [0]))
y_label = tf_idf.sortByKey().map(lambda x : (x[0], [1] if "AU" in str(x[0]) else [0]))
y_pred = y_pred_label.values().reduce(lambda a, b: np.append(a,b))
y = y_label.values().reduce(lambda a, b: np.append(a,b))
diff_pred = y_pred_label.join(y_pred).filter(lambda x: x[1][0] == 1 and x[1][1] == 0).keys()
diff_pred.top(10)


