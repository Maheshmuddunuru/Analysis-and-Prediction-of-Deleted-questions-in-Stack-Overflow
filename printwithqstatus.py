from bs4 import BeautifulSoup
import requests
import re
from csv import writer
import pandas as pd
import itertools

# Methods to check the emppty files
def check_none(file_name):
    if file_name is not None:
        return file_name.text
    else:
        return 0
def check_count(file_name):
    if len(file_name) ==0:
        return 0
    else:
        return file_name[0].text

#Load the Dataset with the Downloaded Binary Files
df = pd.read_csv(r"E:\project papers cite\newTestResults\path3.csv")
df1=pd.DataFrame()
list =[]
list = df.values.tolist()

z=0
def write_to_file(Id, ParentId, CreationDate, Score, Body, OwnerUserId, OwnerDisplayName, Title, Tags, AnswerCount,
                  CommentCount, FavoriteCount, Answerorcomment,clue,reputation_score,viewcount):
    try:
        with open('path3featuretest.csv', "x+", newline="", encoding='utf8') as fout:
            csv_writter = writer(fout)
            csv_writter.writerow(
                [Id, ParentId, CreationDate, Score, Body, OwnerUserId, OwnerDisplayName, Title, Tags, AnswerCount
                    , CommentCount, FavoriteCount, Answerorcomment,clue,reputation_score,viewcount])
            fout.close()
            # Work with your open file
    except FileExistsError:
        # Your error handling goes here
        with open('path3featuretest.csv', 'a+', newline="", encoding='utf8') as my_file:
            csv_writter = writer(my_file)
            csv_writter.writerow(
                [Id, ParentId, CreationDate, Score, Body, OwnerUserId, OwnerDisplayName, Title, Tags, AnswerCount
                    , CommentCount, FavoriteCount, Answerorcomment,clue,reputation_score,viewcount])
            my_file.close()
for i in list:
    try:
        with open(str(i).replace("\\\\","\\").replace("['","").replace("']",""),"r",encoding="utf8") as html_file:
            content= html_file.read()
            #print(content)
        soup = BeautifulSoup(content, 'lxml')


        ##Comments



        # Format the parsed html file
        #print(soup.prettify())
        #question_header = soup.select('.container')

        #post_id = soup.select('.question', id='data-questionid')


        ############################## FIELDS REQUIRED TO FILL IN THE POSTS CSV FILE################################
        answerCount= soup.find("span", itemprop="answerCount")
        comment_id =soup.select('.question .comments .comment',attrs={"id": True})
        #print("ANSWER COUNT IS")
        z=z+1
        if(z%10==0):
            print("Progress is -->",z)
        a=check_none(answerCount)
        #print(a)
        comment_count=0
        for i in comment_id:
            #print(i.get('id').split("-")[1])
            comment_count=comment_count+1

        ########################################################## POSTS SECTIONS######################################################
        ## POSTS FIELD
        #print("HIHIHIHIHI")
        post_id = soup.find_all("div", attrs={"data-questionid": True})
        question_header = soup.select('.question-hyperlink')
        vote_count =soup.select('.vote-count-post')
        body =soup.select('.post-text')
        bodytag=soup.select_one("div.post-text").find_all(['pre'])
        #print("HIHIHIHIHI")

        post =soup.find("div", { "data-questionid" : "" })
        favourite_count = soup.select('.favoritecount')

        user_id = soup.select('.container .post-signature.owner .user-info .user-details a')
        reputation_score=soup.select('.container .post-signature.owner .reputation-score')
        badges =soup.select('.post-signature.owner .user-info .user-details span')
        ## COMMENTED SECTION
        DeletionDate= soup.find('table',id="qinfo")
        ViewCount= DeletionDate.select(".label-key")
        CreationDate= soup.select('.user-action-time .relativetime')
        try:
            qstatus = soup.select('.question-status')
            q=check_count(qstatus)
        except:
            #print("------------------------------------------------------------")
            q=0
            #print(qstatus[0].text)
            #print("------------------------------------------------------------")
            #print("------------------------------------------------------------")

        #DelDate= DeletionDate.select(".label-key .lastactivity-link")
        #print(" QUESTION INFO IS")
        #Deletion Date
        #print(DelDate[0].get('title'))
        #CreationDate
        #print(ViewCount[1].get('title'))
        #VIEW COUNT
        #print(ViewCount[3].text)


        #print(""""HIIIIII""")



        #for i in v:
            #print(i)
        #print("VOTE COUNT IS")
        #vc = soup.sxlect(".question .votecell", attrs={"itemprop": "upvoteCount"})
        #print(vc[0]text.split())
        # POST TITLE
        #print(question_header[0].text)
        # VOTES OF THE QUESTION
        #vote_count=check_count(vote_count)
        vote_count=check_count(vote_count)

        # CREATION DATE OF THE POST
        #print(CreationDate[0].text)
        # BODY OF THE QUESTION
        #print(body[0].text)
        # USER ID OF THE QUESTION
        #print(user_id[0].get('href').split("/")[7])
        # USER ID NAME
        #print("USER IDD DISPLAY NAME")
        #print(user_id[0].text)
        # USER ID REPUTATION SCORE
        print(reputation_score)
        print(reputation_score[0].text)
        if reputation_score is not None:
            repscore=reputation_score[0].text
        else:
            repscore=0
        if ViewCount is not None:
            vcc=ViewCount[3].text
        else:
            vcc=0
        '''
        for i in badges:
            if i.get('title') is not None:
                print(i.get('title'))
        # POSTS Tags
        '''
        tags =soup.select('.post-taglist')
        #print("TAGS OF POST")
        #print(tags[0].text)
        #NUMBER OF FAVOURITES
        #print("Favourite count is ")
        f=check_count(favourite_count)
        #print(f)
        #print(favourite_count[0].text)
        #POST ID OF THE QUESTION
        #print("Print the the post id of the question")
        #print(post_id[0].get('data-questionid'))
        #WRITING EVERYTHING TO THE CSV FILE
        write_to_file(post_id[0].get('data-questionid')," ",CreationDate[0].text,vote_count,(body[0].text),user_id[0].get('href').split("/")[7],user_id[0].text,question_header[0].text,(tags[0].text),check_none(answerCount),comment_count,f ,0,q,repscore,vcc)

        #write_to_file(Id,PostTypeId,AcceptedAnswerId,ParentId,CreationDate,DeletionDate	Score,ViewCount,Body,OwnerUserId,OwnerDisplayName,LastEditorUserId,LastEditorDisplayName,LastEditDate,LastActivityDate,Title,Tags,AnswerCount,CommentCount,FavoriteCount	ClosedDate	CommunityOwnedDate	ContentLicense)
        #write_to_file(post_id[0].get('data-questionid'),1," "," ",CreationDate[0].text,DelDate[0].get('title'),vote_count,ViewCount[3].text,(body[0].text),user_id[0].get('href').split("/")[7],user_id[0].text," "," ", " "," ",question_header[0].text,(tags[0].text),check_none(answerCount),comment_count,f ,"","","",)



        '''print(comment_user_id[0].get('href').split("/")[7])
        comment_user_name =soup.select('.question .comment-user')
        print(comment_user_id[0].get('href').split("/")[8])'''

        ############################################COMMENTS SECIONS####################################################
        #COMMENT FIELDS
        '''
        print("COMMENTS SECTION")
        comment_id =soup.select('.question .comments .comment',attrs={"id": True})
        comments= soup.select('.question .comments .comment .comment-copy')
        comment_votes =soup.select('.question .comments .comment .comment-score span')
        comments_date= soup.select('.question .comments .comment .comment-date')
        comment_user_id = soup.select('.question .comment-user')
        # TO DISPLAY COMMENT USER ID
        for i in comment_user_id:
            print(i.get('href').split("/")[7])
        # TO DISPLAY COMMENT USER Name
        for i in comment_user_id:
            print(i.get('href').split("/")[8])
        # TO DISPLAY COMMENT DATES
        for i in comments_date:
            print("LInting the comments date")
            print(i.text)
        # TO DISPLAY COMMENT COUNT NUMBER and COMMENTES  USER ID's
        print("Listint the comment count and comment post id's")
        comment_count=0
        for i in comment_id:
            print(i.get('id').split("-")[1])
            comment_count=comment_count+1
        print(comment_count)
        # TO DISPLAY THE COMMENTS
        for i in comments:
            print(i.text)
        # TO DISPLAY THE  SCORES OF A COMMENT
        print(" LIsthing the comment scores")
        for i in comment_votes:
            print(i.text)
        # Writing Comments to the csv File:

        '''
        ################################################# ANSWERS SECTION###################################################
        ## ANSWER FIELDS
        #print("ANSWERS ARE")
        answerCount= soup.find("span", itemprop="answerCount")
        answers_id=soup.find_all("div", attrs={"data-answerid": True})
        answers=soup.select(".answercell .post-text")
        answers_votes = soup.find("div", { "id" : "answers" })
        answer_user_id = soup.select(".answercell .post-signature .user-details a")
        answer_dates = soup.select(".answercell .post-signature .user-action-time")
        #print("PRINTING THE ANSWER USER IDS")
        #for i in answer_user_id:
        #    print(i.get('href').split("/")[8])
        #print(answer_user_id)



        #TO DISPLAY THE ANSWER count
        #print("ANSWER COUNT IS")
        #print(answerCount.text)
        # TO DISPPLLAY THE ANSWER POST ID
        #for i in answers_id:
        #    print(i.get('data-answerid'))
        # TO DISPLAY THE VOTES FOR AN ANSWER
        #for ptag in answers_votes.find_all('span', class_='vote-count-post'):
        #    print(ptag.text)
        # TO DISPLAY THE ANSWERS
        #print(answers)
        #for i in answers:
        #    print(i.text)
        #Writing the answers to the  csv File
        for i,ptag,j,k,l in zip(answers_id,(answers_votes.find_all('span', class_='vote-count-post')),answers,answer_user_id,answer_dates):
            write_to_file(i.get('data-answerid'),post_id[0].get('data-questionid'),l.text,ptag.text,j.text,k.get('href').split("/")[7],k.get('href').split("/")[8], " "," "," "," "," ",0,0)

    except:
        continue
df2 = pd.DataFrame(list)
df2.to_csv('rqfinal.csv',index=False)
