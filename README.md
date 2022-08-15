
# Analysis-and-Prediction-of-Deleted-questions-in-Stack-Overflow

![social](https://img.shields.io/github/followers/Maheshmuddunuru?style=social)![twitter](https://img.shields.io/twitter/follow/MaheshK71025493?style=social)
<br/>
This is a README file indicating the use of this repository

## Table of Contents

1. [Title of the Project](#Name-of-the-Project)
2. [Description](#Description)
3. [Languages](#Languages-Used)
4. [Packages](Packages)
5. [Files to run](#Files-to-run)
6. [Usage](#Usage)
7. [Support](#Support)
8. [Future work and Enhancements](#Future-work-and-Enhancements)

## Name of the Project

- Analysis-and-Prediction-of-Deleted-questions-in-Stack-Overflow

## Description
In recent years, Community Question Answering websites (CQAs) 
have grown in popularity as a means of providing and searching 
for information. CQA systems should leverage the collective 
intelligence of the entire online community to maximize their 
effectiveness, which will be impossible without appropriate 
technology support that facilitates collaboration. [Stack Overflow](https://stackoverflow.com/) 
is the largest Community-based Question Answering (CQA) site 
for software developers with over 17 million active users 22M 
question and 33M answers. Stack Overflow provides 
programmers with a forum for asking and answering 
programming-related questions. The Stack Overflow site has
explicit and detailed guidelines for posting questions and an active 
user community that moderates questions. Even with these precise 
communications and guidelines, questions posted on Stack 
Overflow can be extremely off topic and of poor quality. Those 
questions can be deleted at the discretion of moderators and 
community. These deleted questions negatively impact the user 
experience and increase maintenance costs. Because of this, it is 
imperative to analyze and predict deleted questions.
This study consists of two stages (i) Collect deleted questions in 
Stack Overflow, (ii) Develop a predictive model for deletion 
questions at the time of question creation. In the first stage, we 
will collect all the questions and the Internet archive that have been deleted from the stack overflow server. As we create a database of deleted 
questions in the first step, we can then develop a predictive model 
for predicting the deletion of questions the moment they are 
created. In this study our approach is to collect all numerical and meta features that can be categorized into: readability, sentimental, content and syntactic from title, body and tags. Following that, we compare the scores and accuracy of different types of machine learning classifiers over Denzil Correa and Ashish Sureka's baseline [predictions](https://dl.acm.org/doi/10.1145/2566486.2568036) and achieved an overall accuracy of 72.5\% and F1 score 72.4\% improving by 9.93\5 and 10.07\% respectively.

## Languages Used

I have used python Language for this project.

## Packages
This project requires the following packages to run our machine learning [model](https://github.com/Maheshmuddunuru/Analysis-and-Prediction-of-Deleted-questions-in-Stack-Overflow/blob/main/model.py).
```bash 
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
```
The following installation is required to run [Feature Extraction](https://github.com/Maheshmuddunuru/Analysis-and-Prediction-of-Deleted-questions-in-Stack-Overflow/blob/main/delquestioncollection.py) process.
```
pip install textstat
pip3 install readability-lxml
pip install vaderSentiment
python -m nltk.downloader stopwords
```
## Files to run
The collected deleted questions during this study is avaible in the [Google Drive](https://drive.google.com/file/d/1h4eR00gZIGZhmGd5UZKdueArwUQro1b7/view?usp=sharing) for reference. Our Prediction model requires the [deleted feature distribution](https://drive.google.com/file/d/1pT2G0IOFwKq50wdja5LLxMa--81XhQkC/view?usp=sharing) file and the [original feature distribution](https://drive.google.com/file/d/1BA8UG5s33o7Gxe0FCtNpMcaJG5gFNK3j/view?usp=sharing) file in order to be run. 

## Usage
- Once the files are Downloaded, Load the files into prediction model [program](https://github.com/Maheshmuddunuru/Analysis-and-Prediction-of-Deleted-questions-in-Stack-Overflow/blob/main/model.py)
- Load the Model.ipynb file to the notebook.
- Download the dataset WIFIDATAcsv1.csv file to the notebook and adjust the current path of the dataset in the Model.ipynb file.
- Following is a screenshot of the dataset
   <br/>
 ![alt text](https://github.com/Maheshmuddunuru/MITM-Detection-Mechanism-at-client-side-using-Machine-Learning-Algorithms/blob/main/Review%20of%20data.jpg)
   <br/>
- The following is a snippet of code from the project.
   <br/>
 ![alt text](https://github.com/Maheshmuddunuru/MITM-Detection-Mechanism-at-client-side-using-Machine-Learning-Algorithms/blob/main/codesnippet.jpg)
   <br/>
- Make sure all the resources and packages are included and run the code.
- Upon running the code, you will get output with the accuracy.
## Support
You can reach out to me at one of the following places:
- via [email](mmuddunu@lakeheadu.ca)
- via ![twitter](https://img.shields.io/twitter/follow/MaheshK71025493?style=social)
## Future work and enhancements
- There are still a few more areas to test in the dataset.
- Plug and play with the features, Adding Interpretability, Neural Networks and Deep Learning aim at learning feature representations or anomaly scores for the sake of anomaly detection.
- An automated script can also be worked to collect data from the API’s and convert them into csv with a definite structure.
