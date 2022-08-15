import pandas as pd
import numpy as np
from math import log, e
import string
import textstat
import csv
import re
from readability import Readability
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
vader_sentiment_analyzer = SentimentIntensityAnalyzer()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

################1###########################################################DEFINITIONS####################################################
def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
def rem_formatting(text):
    lower = remove_html_tags(text)
    """Removes punctuation from the text and convert into lowercase"""
    punctuationfree="".join([i for i in lower if i not in string.punctuation])
    lower=punctuationfree.lower()
    return(lower)

def remove_stop_words(text):
    """Remove stop words from a string"""
    return [word for word in text if not word in stopwords.words()]
def word_count(raw_content):
    content = rem_formatting(raw_content)
    if(len(content)==0):
        return 0
    else:
        tokenize_content = word_tokenize(content)
        return len(remove_stop_words(tokenize_content))

def pos_sentiment(text, analyzer):
    if(text is None):
        return 0
    score = analyzer.polarity_scores(text)
    return score['pos']

def neg_sentiment(text, analyzer):
    if(text is None):
        return 0
    score = analyzer.polarity_scores(text)
    return score['neg']

def neu_sentiment(text, analyzer):
    if(text is None):
        return 0
    score = analyzer.polarity_scores(text)
    return score['neu']

def comp_sentiment(text, analyzer):
    if(text is None):
        return 0
    score = analyzer.polarity_scores(text)
    return score['compound']
def remtxt(s,startstr,endstr):
        while startstr in s:
                startpos=s.index(startstr)
                try:
                        endpos=s.index(endstr,startpos+len(startstr))+len(endstr)
                except:
                        return
                s=s[:startpos]+s[endpos:]
        return s
def punc(str):
    f=0
    for i in range (0, len(str)):
        #Checks whether given character is a punctuation mark'
        if str[i] in ('!', "," ,"\'" ,";" ,"\"", ".", "-" ,"?"):
            f=f + 1;
    return f
def avgword(str):
    text = str
    length_words = 0
    total_words = 0
    words = text.split( )
    try:
        for x in words:
            length_words += len(x)
            total_words += 1
        average = length_words / total_words
    except:
        average=0
    return(average)
########################################LOAD YOUR FILE INTO THE DATA FRAME########################
df =  pd.read_csv(r"D:\posts.csv",engine='python')
#DROP THE ANSWERS FROM THE DATAFRAME
df = df[df['post_type_id'] ==1]
#Create new Dataframe to Store the metrics
df2= pd.DataFrame()
######################################### ALL CATOEGORY FEATURES INITIALIZATION FOR BODY ###################################

feature_no_of_characters_body=[]
feature_no_of_alphabetic_characters_body=[]
feature_no_of_upper_case_body=[]
feature_no_of_lower_case_body=[]
feature_no_of_digits_body=[]
feature_no_of_white_space_body=[]
feature_no_of_special_char_body=[]
feature_no_of_punctuation_body=[]
feature_no_of_short_words_body=[]
feature_no_of_unique_words_body=[]
feature_no_of_avg_body_length_body=[]
feature_ques_pos_body=[]
feature_ques_neg_body=[]
feature_ques_neu_body=[]
feature_ques_comp_body=[]
feature_body_word_count=[]
feature_flesch_reading_ease_body=[]
feature_flesch_kincaid_grade_body=[]
feature_smog_index_body=[]
feature_coleman_liau_index_body=[]
feature_automated_readability_index_body=[]
feature_dale_chall_readability_score_body=[]
feature_difficult_words_body=[]
feature_linsear_write_formula_body=[]
feature_gunning_fog_body=[]
feature_text_standard_body=[]
feature_fernandez_huerta_body=[]
feature_szigriszt_pazos_body=[]
feature_gutierrez_polini_body=[]
feature_crawford_body=[]
feature_gulpease_index_body=[]
feature_osman_body=[]

################################# ALL CATOEGORY FEATURES INITIALIZATION FOR TITILE############################
feature_no_of_characters_title=[]
feature_no_of_alphabetic_characters_title=[]
feature_no_of_upper_case_title=[]
feature_no_of_lower_case_title=[]
feature_no_of_digits_title=[]
feature_no_of_white_space_title=[]
feature_no_of_special_char_title=[]
feature_no_of_punctuation_title=[]
feature_no_of_short_words_title=[]
feature_no_of_unique_words_title=[]
feature_no_of_avg_word_length_title=[]
feature_ques_pos_title=[]
feature_ques_neg_title=[]
feature_ques_neu_title=[]
feature_ques_comp_title=[]
feature_title_word_count=[]
feature_flesch_reading_ease_body_title=[]
feature_flesch_kincaid_grade_body_title=[]
feature_smog_index_body_title=[]
feature_coleman_liau_index_body_title=[]
feature_automated_readability_index_body_title=[]
feature_dale_chall_readability_score_body_title=[]
feature_difficult_words_body_title=[]
feature_linsear_write_formula_body_title=[]
feature_gunning_fog_body_title=[]
feature_text_standard_body_title=[]
feature_fernandez_huerta_body_title=[]
feature_szigriszt_pazos_body_title=[]
feature_gutierrez_polini_body_title=[]
feature_crawford_body_title=[]
feature_gulpease_index_body_title=[]
feature_osman_body_title=[]
######################################## TAGS FEARURES####################
feature_no_of_tags = []
########################APPENDING QA CONTENT SENTIMENT FOR BODY AND TITLE###############
for i in df.body:
    feature_ques_pos_body.append(pos_sentiment(i,vader_sentiment_analyzer))
    feature_ques_neg_body.append(neg_sentiment(i,vader_sentiment_analyzer))
    feature_ques_neu_body.append(neu_sentiment(i,vader_sentiment_analyzer))
    feature_ques_comp_body.append(comp_sentiment(i,vader_sentiment_analyzer))
    feature_body_word_count.append(word_count(i))
for i in df.title:
    feature_ques_pos_title.append(pos_sentiment(i,vader_sentiment_analyzer))
    feature_ques_neg_title.append(neg_sentiment(i,vader_sentiment_analyzer))
    feature_ques_neu_title.append(neu_sentiment(i,vader_sentiment_analyzer))
    feature_ques_comp_title.append(comp_sentiment(i,vader_sentiment_analyzer))
    feature_title_word_count.append(word_count(i))

##############APPENDING READABILITY VALUES to the Body  and Title #######################################
for i in df.body:
    feature_flesch_reading_ease_body.append(textstat.flesch_reading_ease(i))
    feature_flesch_kincaid_grade_body.append(textstat.flesch_kincaid_grade(i))
    feature_smog_index_body.append(textstat.smog_index(i))
    feature_coleman_liau_index_body.append(textstat.coleman_liau_index(i))
    feature_automated_readability_index_body.append(textstat.automated_readability_index(i))
    feature_dale_chall_readability_score_body.append(textstat.dale_chall_readability_score(i))
    feature_difficult_words_body.append(textstat.difficult_words(i))
    feature_linsear_write_formula_body.append(textstat.linsear_write_formula(i))
    feature_gunning_fog_body.append(textstat.gunning_fog(i))
    feature_fernandez_huerta_body.append(textstat.fernandez_huerta(i))
    feature_szigriszt_pazos_body.append(textstat.szigriszt_pazos(i))
    feature_gutierrez_polini_body.append(textstat.gutierrez_polini(i))
    feature_crawford_body.append(textstat.crawford(i))
    feature_gulpease_index_body.append(textstat.gulpease_index(i))
    feature_osman_body.append(textstat.osman(i))
for i in df.title:
    feature_flesch_reading_ease_body_title.append(textstat.flesch_reading_ease(i))
    feature_flesch_kincaid_grade_body_title.append(textstat.flesch_kincaid_grade(i))
    feature_smog_index_body_title.append(textstat.smog_index(i))
    feature_coleman_liau_index_body_title.append(textstat.coleman_liau_index(i))
    feature_automated_readability_index_body_title.append(textstat.automated_readability_index(i))
    feature_dale_chall_readability_score_body_title.append(textstat.dale_chall_readability_score(i))
    feature_difficult_words_body_title.append(textstat.difficult_words(i))
    feature_linsear_write_formula_body_title.append(textstat.linsear_write_formula(i))
    feature_gunning_fog_body_title.append(textstat.gunning_fog(i))
    feature_fernandez_huerta_body_title.append(textstat.fernandez_huerta(i))
    feature_szigriszt_pazos_body_title.append(textstat.szigriszt_pazos(i))
    feature_gutierrez_polini_body_title.append(textstat.gutierrez_polini(i))
    feature_crawford_body_title.append(textstat.crawford(i))
    feature_gulpease_index_body_title.append(textstat.gulpease_index(i))
    feature_osman_body_title.append(textstat.osman(i))


m=[]
for p in df.title:
    tc=0
    for s in p:
        if(s.isalpha()):
            tc=tc+1
    awl=avgword(p)
    feature_no_of_characters_title.append(tc)
for tags in df.tags:
    j=re.findall("<(.*?)>", tags, re.DOTALL)
    feature_no_of_tags.append(len(j))

for i in df.title:
    alphabetic_characters=0
    lower_case=0
    upper_case=0
    digits=0
    white_space=0
    special_char=0
    punctuation_=0
    wc=0
    short_words=0
    unique_words=0
    avg_word_length=0


    ###############
    punctuation_=punc(i)
    wc= len(i.split())
    lst = re.split(r"'| ", i)
    unique_words=len(set(i))
    avg_word_length=avgword(i)
    short_words_lst = [x for x in lst if x and len(x) <= 3]
    #print(f'Total short words count: {len(short_words_lst)}')
    short_words=len(short_words_lst)
    for s in i:
        if(s.isalpha()):
            alphabetic_characters=alphabetic_characters+1
        else:
             if(s.isdigit()):
                digits=digits+1
             else:
                special_char=special_char+1
        if(s.isupper()):
            upper_caseupper_caseu+1
        if(s.islower()):
            lower_case=lower_case+1
        if(s.isspace()):
            white_space=white_space+1

    feature_no_of_alphabetic_characters_title.append(alphabetic_characters)
    feature_no_of_upper_case_title.append(upper_case)
    feature_no_of_lower_case_title.append(lower_case)
    feature_no_of_digits_title.append(digits)
    feature_no_of_white_space_title.append(white_space)
    feature_no_of_special_char_title.append(special_char)
    feature_no_of_punctuation_title.append(punctuation_)
    feature_no_of_short_words_title.append(short_words)
    feature_no_of_unique_words_title.append(unique_words)
    feature_no_of_avg_word_length_title.append(avg_word_length)

for i in df.body:
    alphabetic_characters=0
    lower_case=0
    upper_case=0
    digits=0
    white_space=0
    special_char=0
    punctuation_=0
    wc=0
    short_words=0
    unique_words=0
    avg_word_length=0


    no_of_characters=len(i)
    punctuation_=punc(i)
    wc= len(i.split())
    lst = re.split(r"'| ", i)
    unique_words=len(set(i))
    avg_body_length=avgword(i)
    short_words_lst = [x for x in lst if x and len(x) <= 3]
    #print(f'Total short words count: {len(short_words_lst)}')
    short_words=len(short_words_lst)
    for s in i:
        if(s.isalpha()):
            c=c+1
        else:
             if(s.isdigit()):
                digits=digits+1
             else:
                special_char=special_char+1
        if(s.isupper()):
            upper_case=upper_case+1
        if(s.islower()):
            lower_case=lower_case+1
        if(s.isspace()):
            white_space=white_space+1

    feature_no_of_characters_body.append(no_of_characters)
    feature_no_of_alphabetic_characters_body.append(alphabetic_characters)
    feature_no_of_upper_case_body.append(white_space)
    feature_no_of_lower_case_body.append(lower_case)
    feature_no_of_digits_body.append(digits)
    feature_no_of_white_space_body.append(white_space)
    feature_no_of_special_char_body.append(special_char)
    feature_no_of_punctuation_body.append(punctuation_)
    feature_no_of_short_words_body.append(short_words)
    feature_no_of_unique_words_body.append(unique_words)
    feature_no_of_avg_body_length_body.append(avg_body_length)


###############BODY METRICS##################

df2['no_of_characters_body']=feature_no_of_characters_body
df2['no_of_alphabetic_characters_body']=feature_no_of_alphabetic_characters_body
df2['no_of_upper_case_body']=feature_no_of_upper_case_body
df2['no_of_lower_case_body']=feature_no_of_lower_case_body
df2['no_of_digits_body']=feature_no_of_digits_body
df2['no_of_white_space_body']=feature_no_of_white_space_body
df2['no_of_special_char_body']=feature_no_of_special_char_body
df2['no_of_punctuation_body']=feature_no_of_punctuation_body
df2['no_of_short_words_body']=feature_no_of_short_words_body
df2['no_of_unique_words_body']=feature_no_of_unique_words_body
df2['no_of_avg_body_length_body']=feature_no_of_avg_body_length_body
df2['ques_pos_body']=feature_ques_pos_body
df2['ques_neg_body']=feature_ques_neg_body
df2['ques_neu_body']=feature_ques_neu_body
df2['ques_comp_body']=feature_ques_comp_body
df2['body_word_count']=feature_body_word_count
df2['flesch_reading_ease_body']=feature_flesch_reading_ease_body
df2['flesch_kincaid_grade_body']=feature_flesch_kincaid_grade_body
df2['smog_index_body']=feature_smog_index_body
df2['coleman_liau_index_body']=feature_coleman_liau_index_body
df2['automated_readability_index_body']=feature_automated_readability_index_body
df2['dale_chall_readability_score_body']=feature_dale_chall_readability_score_body
df2['difficult_words_body']=feature_difficult_words_body
df2['linsear_write_formula_body']=feature_linsear_write_formula_body
df2['gunning_fog_body']=feature_gunning_fog_body
df2['fernandez_huerta_body']=feature_fernandez_huerta_body
df2['szigriszt_pazos_body']=feature_szigriszt_pazos_body
df2['gutierrez_polini_body']=feature_gutierrez_polini_body
df2['crawford_body']=feature_crawford_body
df2['gulpease_index_body']=feature_gulpease_index_body
df2['osman_body']=feature_osman_body
############################# TITLE METRICS#############################################
df2['no_of_characters_title']=feature_no_of_characters_title
df2['no_of_alphabetic_characters_title']=feature_no_of_alphabetic_characters_title
df2['no_of_upper_case_title']=feature_no_of_upper_case_title
df2['no_of_lower_case_title']=feature_no_of_lower_case_title
df2['no_of_digits_title']=feature_no_of_digits_title
df2['no_of_white_space_title']=feature_no_of_white_space_title
df2['no_of_special_char_title']=feature_no_of_special_char_title
df2['no_of_punctuation_title']=feature_no_of_punctuation_title
df2['no_of_short_words_title']=feature_no_of_short_words_title
df2['no_of_unique_words_title']=feature_no_of_unique_words_title
df2['no_of_avg_word_length_title']=feature_no_of_avg_word_length_title
df2['ques_pos_title']=feature_ques_pos_title
df2['ques_neg_title']=feature_ques_neg_title
df2['ques_neu_title']=feature_ques_neu_title
df2['ques_comp_title']=feature_ques_comp_title
df2['title_word_count']=feature_title_word_count
df2['flesch_reading_ease_body_title']=feature_flesch_reading_ease_body_title
df2['flesch_kincaid_grade_body_title']=feature_flesch_kincaid_grade_body_title
df2['smog_index_body_title']=feature_smog_index_body_title
df2['coleman_liau_index_body_title']=feature_coleman_liau_index_body_title
df2['automated_readability_index_body_title']=feature_automated_readability_index_body_title
df2['dale_chall_readability_score_body_title']=feature_dale_chall_readability_score_body_title
df2['difficult_words_body_title']=feature_difficult_words_body_title
df2['linsear_write_formula_body_title']=feature_linsear_write_formula_body_title
df2['gunning_fog_body_title']=feature_gunning_fog_body_title
df2['fernandez_huerta_body_title']=feature_fernandez_huerta_body_title
df2['szigriszt_pazos_body_title']=feature_szigriszt_pazos_body_title
df2['gutierrez_polini_body_title']=feature_gutierrez_polini_body_title
df2['crawford_body_title']=feature_crawford_body_title
df2['gulpease_index_body_title']=feature_gulpease_index_body_title
df2['osman_body_title']=feature_osman_body_title
#####################################
df2['no_of_tags'] = feature_no_of_tags


##################  BODY METRICS##############################
df2.to_csv('oroginalfeaturedistribution.csv',index=False)
