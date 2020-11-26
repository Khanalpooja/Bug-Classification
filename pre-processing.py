
# A. 1. importing the libraries
import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
import matplotlib.pyplot as plt # used for ploting the data into python
import matplotlib.patches as mpatches #for setting legend colors
import sklearn
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import regexp_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from csv import reader
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim 
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from models import models
import models as md
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.pipeline import Pipeline
from skmultilearn.dataset import load_dataset
from sklearn.multioutput  import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
# from skmultilearn.multilearn.model_selection.measures import get_combination_wise_output_matrix

import csv

# C. 1. Importing libraries for model selection matrix
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from nltk.stem import WordNetLemmatizer 
from num2words import num2words
nltk.download('wordnet')
# D. 1. Importing libraries for clean output 
from warnings import simplefilter# import warnings filter
simplefilter(action='ignore', category=FutureWarning)# ignore all future warnings

# 2. Pre-processing 
# Pre-processing of Text body - title is not taken into consideration for now
lb_make = LabelEncoder()
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) 
filtered_sentence = []

# 2.1 Tokenization
# ignore_encoding = lambda s: s.decode('utf8', 'ignore')
# csv_csv = pd.read_csv("output.csv")
# reader = csv.reader(x.replace('\0', '') for x in csv_csv

# print(csv_csv.shape)
with open("output_new_3.csv", "r", encoding ="ISO-8859-1") as read_obj:
    # q = read_obj.read()
    # r = q.decode()
    csv.reader((line.replace('\0','') for line in read_obj) )

    # pass the file object to reader() to get the reader object
    csv_reader =  csv.reader((line.replace('\0','') for line in read_obj) )
    csv.field_size_limit(1000000000000)

    csv_reader = reader(read_obj)
    next(csv_reader, None)  # skip the headers
    # Iterate over each row in the csv using reader object
    # need to by pass the header
    count = 0
    whole_list=[]
    label_list=[]
    joined_list=[]
    count = 0
    for row in csv_reader:
        
    
        # row variable is a list that represents a row in csv for taking the body content
        # Taking labels 
        

        #Handling missing dataset 

        label_content = row[5]
        body_content = row[1]
        # print(label_content)
        # print(body_content)
        if label_content != "[]" and label_content != '' and body_content != '' and label_content != '\0' and body_content != '\0' and label_content != 'NUL' and body_content != 'NUL' and label_content.startswith('['):
            

            # print(label_content)
            stemmed_list=[]
            lemmitized_list=[]
            lower_case_list=[]
            num_to_words_list=[]
            tokenized = regexp_tokenize(body_content, "[\w']+")
            #Removing punctuation 
            tokenized_punctuation_removed= [word for word in tokenized if word.isalnum()]
            for each in tokenized_punctuation_removed:
                #Removing default stop words
                if each not in stop_words: 
                    filtered_sentence.append(each) 
                    # Normalization
                    # 2.2 Stemming 
                    stemmed = ps.stem(each)
                    stemmed_list.append(stemmed)
                    # 2.3 Lemmatization
                    lemmitized =lemmatizer.lemmatize(stemmed)
                    lemmitized_list.append(lemmitized) 
                    # 2.4 Comverting to lower case
                    lower_case_list.append(lemmitized.lower())
                    #2.5 converting number to words
                    # if lemmitized.lower().isnumeric() == True :
                    #     num_to_word = num2words(lemmitized.lower())
                    # else:
                    num_to_word = lemmitized.lower()
                    num_to_words_list.append(num_to_word)
                    # print(num_to_words_list)
                
            # print(label_content)
            label_content_new = label_content.split("'")[1]
            

            
            if label_content_new.lower() == "bug":
                
                if len(label_content.split(",")) >=3:
                    count +=1
                    label_content_second = label_content.split("'")[3]
                    print(label_content_second)
                    print(label_content)
                    # for i in range(3, len(label_content.split("'"))):
                    #     # print("lalalala")
                    #     if i%2 != 0:
                    #         # print(label_content.split("'")[i])
                    #         label_list.append(label_content.split("'")[i])

 
            else: 
                count +=1
                
                label_content_second = label_content_new
            label_list.append(label_content_second)

            joined_words = " ".join(num_to_words_list)

            whole_list.append(num_to_words_list)
            joined_list.append(joined_words)
            num_to_words_list.sort()





        multilabel_binarizer = MultiLabelBinarizer()
        multilabel_binarizer.fit(label_content)
        Y_la = multilabel_binarizer.transform(label_content)
        print(label_content)
        print(Y_la)

        # count_vect = CountVectorizer()
        # body_content = [body_content]
        # X_counts = count_vect.fit_transform(body_content)

        # tfidf_transformer = TfidfTransformer()
        # X_tfidf = tfidf_transformer.fit_transform(X_counts)

        # ros = RandomOverSampler(random_state=9000)
        # X_tfidf_resampled, Y_tfidf_resampled = ros.fit_sample(X_tfidf, Y_la)  
        # x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf_resampled, Y_tfidf_resampled, test_size=0.2, random_state=9000) 
        # print(x_train_tfidf)
        # print(x_test_tfidf)
        # print(y_train_tfidf)
        # print(y_test_tfidf)


        # if count >= 2000:
        #     break




    print(label_list)
    for i in range(len(label_list)):
        # print(label_list[i])
        label = label_list[i]
        if label.startswith("IO") or label=="Output-Formatting":
            label_list[i] = "IO"
        elif label == "Window"   or label == "Typing" or label == "Windows" or label=="Reshaping" or label=="Visualization" or label=="Style" or label=="Styler" :
            label_list[i] = "Visualization"
        elif label.find("Index") > -1 or label.find("Array") > -1 or label.find("Dtypes") > -1 or label.find("Numeric") > -1 or label == "DataFrame" or label=="Missing-data" or label=="Strings" or label=="Categorical" or label=="Period" or label=="Resample" or label=="Interval" or label=="Groupby":
            label_list[i] = "DataStructure"
        elif label== "Needs Discussion" or label == "Unreliable Test" or label == "Closing Candidate" or label == "Usage Question" or label=="Needs Info" or label=="Needs Tests" or label=="Needs Triage" or label=="Testing" or label=="Regression" or label=="Duplicate" or label=="Docs" or label=="Error Reporting" or label=="Compat" or label=="Constructors" or label=="Unreliabale Test" or label=="Dependencies":
            label_list[i] = "Misc"
        elif label=="Timezone" or label=="Timezones" or label=="Timeseries" or label=="Frequency" or label=="Timedelta":
            label_list[i] = "DateTime"
        elif label == "API - Consistency" or label=="API Design":
            label_list[i] = "API"
        elif label =="Benchmark":
            label_list[i] = "Performance"
        
        # print(label)

    print(count)
    print("------------------------") 
    print(label_list)
    # unique values in the list
    setu = set(label_list)
    print(setu)
    print("-----------------------")
    print(len(setu))



    # print(len(joined_list))
    # 2.6 : Bugs synonym dictionary 
    # Bugs Synonym dictionary 
    # UI : User Interface
    # Syntax : Syntatic
    # Network : Communication, Connection
    # Security : Vulnerability 

    bug_type_dict = {"UI": ["User Interface"], "Syntax": ["syntatic"], "Network": ["Communication", "Connection"], "Security" : ["vulnerability"]}
    # print(bug_type_dict)
    # 3.1 Combine and create a word2vec model 
    # # Take some sample sentences
    # Initialise model, for more information, please check the Gensim Word2vec documentation
    # model = Word2Vec(joined_list, size=10, window=15, min_count=2, iter=10)

    # Get a list of words in the vocabulary
    # words = model.wv.vocab.keys()
    # Make a dictionary
    # we_dict = {word:model.wv[word] for word in words}
    # print("Whole list")
    # print(whole_list)
    # print("vocab")
    # print(words)
    # print("label_list")
    # print(label_list)
    # w1 = "bug"
    # print(model.wv.most_similar(positive=w1))

    #Handling of categorical data 
    Y = lb_make.fit_transform(label_list)
    # print(Y)
    # A. 5. Splitting the dataset into training and testing dataset
    # train, test = train_test_split(joined_list, random_state=42, test_size=0.33, shuffle=True)


    X_train, X_test, Y_train, Y_test = train_test_split(joined_list, Y, test_size = 0.25, random_state = 42)


    # print(Y_test)
    # print(X_train)
    # print("------------")
    # print(X_test)
    # print("-----------")
    # print(Y_train)
    # print("-----------")
    # print(Y_test)


    # A.  6. Feature Scaling
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train) #normalizing data with a particular range 
    # X_test = sc.transform(X_test) #add-sub same range for test data


    vectorizer, vocab, train_data_features, tfidf_features, tfidf  = md.create_bag_of_words(X_train)

    # bag_dictionary = pd.DataFrame()
    # bag_dictionary['ngram'] = vocab
    # bag_dictionary['count'] = train_data_features[0]
    # bag_dictionary['tfidf_features'] = tfidf_features[0]

    # Sort by raw count
    # bag_dictionary.sort_values(by=['count'], ascending=False, inplace=True)
    # Show top 10
    # print(bag_dictionary.head(10))




    new_models = models()
    ml_model_lr = new_models.train_logistic_regression(tfidf_features, Y_train)
    ml_model_knn = new_models.train_KNN(tfidf_features, Y_train)
    ml_model_g = new_models.train_gaussian(tfidf_features, Y_train)
    ml_model_svm = new_models.train_svm(tfidf_features, Y_train)
    ml_model_svm_kernel = new_models.train_gaussian(tfidf_features, Y_train)
    ml_model_dt = new_models.train_svm(tfidf_features, Y_train)
    ml_model_rf = new_models.train_svm(tfidf_features, Y_train)


    # Common for all 
    test_data_features = vectorizer.transform(X_test)
    test_data_features = test_data_features.toarray()
    test_data_tfidf_features = tfidf.fit_transform(test_data_features)
    test_data_tfidf_features = test_data_tfidf_features.toarray()

    predicted_y = ml_model_lr.predict(test_data_tfidf_features)
    predicted_y_knn = ml_model_lr.predict(test_data_tfidf_features)
    predicted_y_g = ml_model_g.predict(test_data_tfidf_features)
    predicted_y_svm = ml_model_svm.predict(test_data_tfidf_features)
    predicted_y_svm_kernel = ml_model_svm_kernel.predict(test_data_tfidf_features)
    predicted_y_rf = ml_model_rf.predict(test_data_tfidf_features)
    predicted_y_dt = ml_model_dt.predict(test_data_tfidf_features)

    # NB_pipeline = Pipeline([
    #     ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    #     ('clf', OneVsRestClassifier(MultinomialNB(
    #         fit_prior=True, class_prior=None))),
    # ])
    # for category in label_list:
    #     print('... Processing {}'.format(category))
    #     # train the model using X_dtm & y
    #     NB_pipeline.fit(X_train, Y_train)
    #     # compute the testing accuracy
    #     prediction = NB_pipeline.predict(X_test)
    #     print('Test accuracy is {}'.format(accuracy_score(X_test, prediction)))

    matrix = md.matrix()
    print("Accuracy-----------------")
    matrix.accuracy("Gaussian", predicted_y_g, Y_test) 
    matrix.accuracy("KNN", predicted_y_knn, Y_test)
    matrix.accuracy("LR", predicted_y, Y_test)
    matrix.accuracy("SVM", predicted_y_svm, Y_test)
    matrix.accuracy("SVM kernel", predicted_y_svm_kernel, Y_test)
    matrix.accuracy("Decision Tree", predicted_y_dt, Y_test)
    matrix.accuracy("Random Forest", predicted_y_rf, Y_test)


    print("Recall-------------------")
    matrix.recall("Gaussian", predicted_y_g, Y_test) 
    matrix.recall("KNN", predicted_y_knn, Y_test)
    matrix.recall("LR", predicted_y, Y_test)
    matrix.recall("SVM", predicted_y_svm, Y_test)
    matrix.recall("SVM kernel", predicted_y_svm_kernel, Y_test)
    matrix.recall("Decision Tree", predicted_y_dt, Y_test)
    matrix.recall("Random Forest", predicted_y_rf, Y_test)

    print("Precision----------------------")
    matrix.precision("Gaussian", predicted_y_g, Y_test) 
    matrix.precision("KNN", predicted_y_knn, Y_test)
    matrix.precision("LR", predicted_y, Y_test)
    matrix.precision("SVM", predicted_y_svm, Y_test)
    matrix.precision("SVM kernel", predicted_y_svm_kernel, Y_test)
    matrix.precision("Decision Tree", predicted_y_dt, Y_test)
    matrix.precision("Random Forest", predicted_y_rf, Y_test)


    print("F1-Score----------------------")
    matrix.f1("Gaussian", predicted_y_g, Y_test) 
    matrix.f1("KNN", predicted_y_knn, Y_test)
    matrix.f1("LR", predicted_y, Y_test)
    matrix.f1("SVM", predicted_y_svm, Y_test)
    matrix.f1("SVM kernel", predicted_y_svm_kernel, Y_test)
    matrix.f1("Decision Tree", predicted_y_dt, Y_test)
    matrix.f1("Random Forest", predicted_y_rf, Y_test)



















