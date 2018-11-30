import re
import numpy as np
from collections import defaultdict

# Thomas Herring
# Comp 330 Assignment 4

#Lab 5

corpus = sc.textFile ("s3://chrisjermainebucket/comp330_A6/20_news_same_line.txt")

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


# Part 1

word_docs = keyAndListOfWords.flatMap(lambda x: ((word, str(x[0])) for word in x[1]))

temp = word_dict.join(word_docs)
temp.groupByKey().map(lambda x: (x[0], list(x[1])))
doc_ranks = temp.map(lambda x: (x[1][1], x[1][0]))

# Convert the current python array into a numpy data structure
def npArrayConvert(list1):
    array = np.zeros(20000)
    for thing in list1:
        array[thing] += 1
    return array

rdd_map = doc_ranks.groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda x: (x[0], npArrayConvert(x[1])))


# Show results on console
a = rdd_map.lookup("20_newsgroups/comp.graphics/37261")[0]
a[a.nonzero()]

b = rdd_map.lookup("20_newsgroups/talk.politics.mideast/75944")[0]
b[b.nonzero()]

c = rdd_map.lookup("20_newsgroups/sci.med/58763")[0]
c[c.nonzero()]

# Part 2

tf = rdd_map.map(lambda x: (x[0], x[1]/np.sum(x[1])))

num_docs = keyAndListOfWords.count()

# Sets elements to 1 if word is in document
def inDocument(vector):
    for i in range(20000):
        if vector[i] > 1:
            vector[i] = 1
    return vector

# Mapping with number of appearences of words in docs
word_in_doc = rdd_map.map(lambda x: (1, inDocument(x[1])))

# Finds total bumber of docs in which words appear in and then reduces by key
all_words_in_doc = word_in_doc.aggregateByKey(np.zeros(20000), lambda a, b: a+b, lambda a,b: a+b)
denominator = all_words_in_doc.top(1)[0][1]

idf = np.log(num_docs/denominator)

def vectorMult(tf,idf):
    tf_idf1 = np.zeros(20000)
    for num in range(20000):
        tf_idf1[num] = tf[num]*idf[num]
    return tf_idf1

tf_idf = tf.map(lambda x: (x[0], vectorMult(x[1], idf)))

a = tf_idf.lookup("20_newsgroups/comp.graphics/37261")[0]
a[a.nonzero()]

b = tf_idf.lookup("20_newsgroups/talk.politics.mideast/75944")[0]
b[b.nonzero()]

c = tf_idf.lookup("20_newsgroups/sci.med/58763")[0]
c[c.nonzero()]


# Part 3

def elementWiseSubtract(item1, item2):
    subtracted_array = np.zeros(20000)
    for i in range(20000):
        subtracted_array[i] = item1[i]-item2[i]
    return subtracted_array

def predictLabel(num_k, text):
    
    words = []
    for word in re.split('\W+', text):
        words.append(word.lower())

    words_rdd = sc.parallelize(words)
    correct_idx = word_dict.join(words_rdd.map(lambda x: (unicode(x),1)))

    word_position = correct_idx.map(lambda x: (x[1][1], x[1][0]))
    position_list = word_position.groupByKey().map(lambda x: (x[0], list(x[1])))
    count_vector = position_list.map(lambda x: (x[0], npArrayConvert(x[1])))

    # Count vector for output
    tf_out = count_vector.map(lambda x: (x[0], x[1]/np.sum(x[1])))
    tf_out_idf = tf_out.map(lambda x: (x[0], vectorMult(x[1], idf)))

    # Ensure there wasn't a pickling error
    tf_array = tf_out_idf.top(1)[0][1]
    subtracted = tf_idf.map(lambda x: (x[0], elementWiseSubtract(x[1],tf_array)))

    norms = subtracted.map(lambda x: (x[0], np.linalg.norm(x[1])))
    possible = norms.takeOrdered(num_k, key=lambda x: x[1])

    # Create mapping of topics to number of occurances in top k
    possible_answers = []
    for i in range(len(possible)):
        possible_answers[i] = (possible[i][0].split('/')[1], possible[i][1])

    topics_mapping = defaultdict(int)
    for item in possible_answers:
        topics_mapping[item[0]] += 1


    return max(topics_mapping, key=topics_mapping.get)



predictLabel (10, 'Graphics are pictures and movies created using computers – usually referring to image data created by a computer specifically with help from specialized graphical hardware and software. It is a vast and recent area in computer science. The phrase was coined by computer graphics researchers Verne Hudson and William Fetter of Boeing in 1960. It is often abbreviated as CG, though sometimes erroneously referred to as CGI. Important topics in computer graphics include user interface design, sprite graphics, vector graphics, 3D modeling, shaders, GPU design, implicit surface visualization with ray tracing, and computer vision, among others. The overall methodology depends heavily on the underlying sciences of geometry, optics, and physics. Computer graphics is responsible for displaying art and image data effectively and meaningfully to the user, and processing image data received from the physical world. The interaction and understanding of computers and interpretation of data has been made easier because of computer graphics. Computer graphic development has had a significant impact on many types of media and has revolutionized animation, movies, advertising, video games, and graphic design generally.')

predictLabel (10, 'A deity is a concept conceived in diverse ways in various cultures, typically as a natural or supernatural being considered divine or sacred. Monotheistic religions accept only one Deity (predominantly referred to as God), polytheistic religions accept and worship multiple deities, henotheistic religions accept one supreme deity without denying other deities considering them as equivalent aspects of the same divine principle, while several non-theistic religions deny any supreme eternal creator deity but accept a pantheon of deities which live, die and are reborn just like any other being. A male deity is a god, while a female deity is a goddess. The Oxford reference defines deity as a god or goddess (in a polytheistic religion), or anything revered as divine. C. Scott Littleton defines a deity as a being with powers greater than those of ordinary humans, but who interacts with humans, positively or negatively, in ways that carry humans to new levels of consciousness beyond the grounded preoccupations of ordinary life.')

predictLabel (10, 'Egypt, officially the Arab Republic of Egypt, is a transcontinental country spanning the northeast corner of Africa and southwest corner of Asia by a land bridge formed by the Sinai Peninsula. Egypt is a Mediterranean country bordered by the Gaza Strip and Israel to the northeast, the Gulf of Aqaba to the east, the Red Sea to the east and south, Sudan to the south, and Libya to the west. Across the Gulf of Aqaba lies Jordan, and across from the Sinai Peninsula lies Saudi Arabia, although Jordan and Saudi Arabia do not share a land border with Egypt. It is the worlds only contiguous Eurafrasian nation. Egypt has among the longest histories of any modern country, emerging as one of the worlds first nation states in the tenth millennium BC. Considered a cradle of civilisation, Ancient Egypt experienced some of the earliest developments of writing, agriculture, urbanisation, organised religion and central government. Iconic monuments such as the Giza Necropolis and its Great Sphinx, as well the ruins of Memphis, Thebes, Karnak, and the Valley of the Kings, reflect this legacy and remain a significant focus of archaeological study and popular interest worldwide. Egypts rich cultural heritage is an integral part of its national identity, which has endured, and at times assimilated, various foreign influences, including Greek, Persian, Roman, Arab, Ottoman, and European. One of the earliest centers of Christianity, Egypt was Islamised in the seventh century and remains a predominantly Muslim country, albeit with a significant Christian minority.')

predictLabel (10, 'The term atheism originated from the Greek atheos, meaning without god(s), used as a pejorative term applied to those thought to reject the gods worshiped by the larger society. With the spread of freethought, skeptical inquiry, and subsequent increase in criticism of religion, application of the term narrowed in scope. The first individuals to identify themselves using the word atheist lived in the 18th century during the Age of Enlightenment. The French Revolution, noted for its unprecedented atheism, witnessed the first major political movement in history to advocate for the supremacy of human reason. Arguments for atheism range from the philosophical to social and historical approaches. Rationales for not believing in deities include arguments that there is a lack of empirical evidence; the problem of evil; the argument from inconsistent revelations; the rejection of concepts that cannot be falsified; and the argument from nonbelief. Although some atheists have adopted secular philosophies (eg. humanism and skepticism), there is no one ideology or set of behaviors to which all atheists adhere.')

predictLabel (10, 'President Dwight D. Eisenhower established NASA in 1958 with a distinctly civilian (rather than military) orientation encouraging peaceful applications in space science. The National Aeronautics and Space Act was passed on July 29, 1958, disestablishing NASAs predecessor, the National Advisory Committee for Aeronautics (NACA). The new agency became operational on October 1, 1958. Since that time, most US space exploration efforts have been led by NASA, including the Apollo moon-landing missions, the Skylab space station, and later the Space Shuttle. Currently, NASA is supporting the International Space Station and is overseeing the development of the Orion Multi-Purpose Crew Vehicle, the Space Launch System and Commercial Crew vehicles. The agency is also responsible for the Launch Services Program (LSP) which provides oversight of launch operations and countdown management for unmanned NASA launches.')

predictLabel (10, 'The Colt Single Action Army which is also known as the Single Action Army, SAA, Model P, Peacemaker, M1873, and Colt .45 is a single-action revolver with a revolving cylinder holding six metallic cartridges. It was designed for the U.S. government service revolver trials of 1872 by Colts Patent Firearms Manufacturing Company – todays Colts Manufacturing Company – and was adopted as the standard military service revolver until 1892. The Colt SAA has been offered in over 30 different calibers and various barrel lengths. Its overall appearance has remained consistent since 1873. Colt has discontinued its production twice, but brought it back due to popular demand. The revolver was popular with ranchers, lawmen, and outlaws alike, but as of the early 21st century, models are mostly bought by collectors and re-enactors. Its design has influenced the production of numerous other models from other companies.')

predictLabel (10, 'Howe was recruited by the Red Wings and made his NHL debut in 1946. He led the league in scoring each year from 1950 to 1954, then again in 1957 and 1963. He ranked among the top ten in league scoring for 21 consecutive years and set a league record for points in a season (95) in 1953. He won the Stanley Cup with the Red Wings four times, won six Hart Trophies as the leagues most valuable player, and won six Art Ross Trophies as the leading scorer. Howe retired in 1971 and was inducted into the Hockey Hall of Fame the next year. However, he came back two years later to join his sons Mark and Marty on the Houston Aeros of the WHA. Although in his mid-40s, he scored over 100 points twice in six years. He made a brief return to the NHL in 1979–80, playing one season with the Hartford Whalers, then retired at the age of 52. His involvement with the WHA was central to their brief pre-NHL merger success and forced the NHL to expand their recruitment to European talent and to expand to new markets.')

