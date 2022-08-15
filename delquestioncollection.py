# Importing Libraries
import requests
from stem import Signal
from stem.control import Controller
import pandas as pd
import os,sys
import time
import waybackpack
import logging
import csv
from csv import writer
DEFAULT_ROOT = "https://web.archive.org"
DEFAULT_USER_AGENT = "waybackpack"
logger = logging.getLogger(__name__)


# Reading the Data File
df = pd.read_csv(r"E:\project papers cite\Stack_Overflow_DelatedPosts - Copy\query.csv")
# Storing the url structure into a list
my_new_list = ['https://stackoverflow.com/q/'+ str(x) for x in df.Id]

#Definition to Download the files if any snapshot is available
def download(self, directory,
        raw=False,
        root=DEFAULT_ROOT,
        ignore_errors=False):
        n=0
        n=0
        for asset in self.assets:
            #Downloading only one snapshot
            n=n+1
            if(n>2):
                break
            else:
                path_head, path_tail = os.path.split(self.parsed_url.path)
                if path_tail == "":
                    path_tail = "index.html"
                if (len(path_head.split("/")) < 3):
                    continue
                else:
                    filedir = os.path.join(
                        directory,
                        path_head.split("/")[2]
                    )

                    filepath = os.path.join(filedir, path_tail)

                    logger.info(
                        "Fetching {0} @ {1}".format(
                            asset.original_url,
                            asset.timestamp)
                    )

                    try:
                        content = asset.fetch(
                            session=self.session,
                            raw=raw,
                            root=root
                        )

                        if content is None:
                            continue

                    except Exception as e:
                        if ignore_errors == True:
                            ex_name = ".".join([ e.__module__, e.__class__.__name__ ])
                            logger.warn("ERROR -- {0} @ {1} -- {2}: {3}".format(
                                asset.original_url,
                                asset.timestamp,
                                ex_name,
                                e
                            ))
                            continue
                        else:
                            raise

                    try:
                        os.makedirs(filedir)
                    except OSError:
                        pass

                    with open(filepath, "wb") as f:
                        logger.info("Writing to {0}\n".format(filepath))
                        f.write(content)
                        
#Definition for storing the  question number, title, snapshots and the link
def write_to_file(question,title,snapshots,link):
    try:
        with open("loggingfile.csv", "x+",newline="") as fout:
          csv_writter = writer(fout)
          csv_writter.writerow([question,title,snapshots,link])
          fout.close()
         #Work with your open file
    except FileExistsError:
      # Your error handling goes here
      with open ('loggingfile.csv','a+',newline="") as my_file:
         csv_writter = writer(my_file)
         csv_writter.writerow([question,title,snapshots,link])
         my_file.close()

#Generation TOR session
def get_tor_session():
    session = requests.session()
    # Tor uses the 9050 port as the default socks port
    session.proxies = {'http':  'socks5://127.0.0.1:9050',
                       'https': 'socks5://127.0.0.1:9050'}
    return session

# Make a request through the Tor connection
# IP visible through Tor
session = get_tor_session()
print(session.get("http://httpbin.org/ip").text)
# Above should print an IP different than your public IP

# Following prints your normal public IP
#print(requests.get("http://httpbin.org/ip").text)
from stem import Signal
from stem.control import Controller

# signal TOR for a new connection
def renew_connection():
    with Controller.from_port(port = 9051) as controller:
        controller.authenticate(password="coolkid")
        controller.signal(Signal.NEWNYM)

#Creating a new list to store the Title
title = []
renew_connection()
session = get_tor_session()
print(session.get("http://httpbin.org/ip").text)
renew_connection()
print(session.get("http://httpbin.org/ip").text)
#Creating the target list to store the sessions of a queried URL
target =[]
c=0
m=0
val = input("Specify the row number where your program should run: ")

for i in range(int(val),len(my_new_list)):

    renew_connection()
    session = get_tor_session()
    try:
        target.append(session.head(my_new_list[i], allow_redirects=True).url)
        snapshots = waybackpack.search(target[c],uniques_only=True)
        timestamps = [snap["timestamp"] for snap in snapshots]
        packed_results = waybackpack.Pack(target[c], timestamps)
        print(target[c])
        #Storing the post id, title, number of snapshots and link
        if (len(target[c].split("/")) > 5):
            #print(target[c].split("/")[4])
            title = target[c].split("/")[5].split("-")
            title=" ".join(title)
            write_to_file(target[c].split("/")[4],title,len(packed_results.assets),target[c])
        else:
            write_to_file(target[c].split("/")[4]," ",0,target[c])


        # print("Original URL is %s",title)

        #Variable to run only for 1 snapshot
        m=0
        m=0
        for asset in packed_results.assets:
            # get the location of the archived URL
            archive_url = asset.get_archive_url()
            m=m+1
            if(m>1):
                break
            else:
                print ("[*] Retrieving %s ( of %d)" % (archive_url,len(packed_results.assets)))
                #   grab the HTML from the Wayback machine
                result1 = asset.fetch()
                #DOwnloadint the archieved files to Directory file
                files = download(packed_results,"NewResults")
        session.close()
        session = get_tor_session()
        print(i)
        print(target[c])
        c=c+1
        #Suspending the program for 10 seconds
        time.sleep(10)
    except requests.ConnectionError as e:
        print("OOPS!! Connection Error. Make sure you are connected to Internet. Technical Details given below.\n")
        print(str(e))
        renew_connection()
        continue
    except requests.Timeout as e:
        print("OOPS!! Timeout Error")
        print(str(e))
        renew_connection()
        continue
    except requests.RequestException as e:
        print("OOPS!! General Error")
        print(str(e))
        renew_connection()
        continue
    except KeyboardInterrupt:
        print("Someone closed the program")
