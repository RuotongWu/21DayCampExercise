
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def read_data():

    '''
    read data
    '''
    path = os.getcwd() + "\\Data\\employees_dataset.csv"
    data = pd.read_csv(path)

    return data

def convert_to_digits(dataframe,list_feature_list,tags):

    '''
    Using word vectorizer convert the dataset to vectors
    idea: Bag of word
    '''

    for features in list_feature_list:
        dataframe[str(features) + '_space'] = [string.title() for string in dataframe[features]]
        dataframe[str(features) + '_space'] = [string.strip().replace(" ", "-") for string in dataframe[str(features) + '_space']]
        dataframe[str(features) + '_space'] = [string.strip().replace(";", " ") for string in dataframe[str(features) + '_space']]
        
        vectorizer = CountVectorizer(token_pattern=r"\w\S+\w")
        feature_list = vectorizer.fit_transform(dataframe[str(features) + '_space'])
        vectorizer.get_feature_names()
        result = feature_list.toarray().tolist()

    

        dataframe[str(features) + '_converted'] = result

    '''
    convert the tags into digits
    '''
    tag_list = dataframe[tags]
    for index, tag in enumerate(set(tag_list)):
        tag_list = np.where(tag_list == tag, index, tag_list)

    return dataframe, tag_list




if __name__ == "__main__":
    '''
    read data
    '''
    dataframe = read_data()
    dataframe = dataframe.sample(frac = 1).reset_index(drop = True)
    list_feature_list = ["degree","education","skills","working_experience"]
    tags = "position"

    '''
    convert to digits
    '''
    data,tag = convert_to_digits(dataframe,list_feature_list,tags)
    train = data[["degree_converted","education_converted","skills_converted","working_experience_converted","position"]]

    tag = tag.astype('int')
    degree=data["degree_converted"].values
    education = data["education_converted"].values
    skills = dataframe["skills_converted"].values
    working_experience = data["working_experience_converted"].values

    '''
    combine data
    '''
    train_data = []
    for d,e,s,w in zip(degree,education,skills,working_experience):
        train_data.append(d+e+s+w)
    train_data
    training_data,test_data, training_tags, test_tags = train_test_split(train_data,tag,test_size = 0.25)
    lr = LogisticRegression()
    lr.fit(training_data,training_tags)
    test_predict = lr.predict(test_data)
    print(classification_report(test_tags,test_predict))