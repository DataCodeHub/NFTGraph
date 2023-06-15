from codecs import ignore_errors
import datetime
from inspect import FullArgSpec
from sqlite3 import Timestamp
# from types import NoneType
# from types import NoneType
import requests
# from retrying import retry  # 需第三方库，需pip进行安装
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os
import pandas as pd
import csv
from bs4 import BeautifulSoup
import time
# from lxml import etree
import re

# @retry(wait_fixed=5, stop_max_attempt_number=10)
def click(path):
    driver.find_element(By.XPATH, path).click()

# @retry(wait_fixed=5, stop_max_attempt_number=10)
def get(url):
    tx_html = driver.get(url)
    return tx_html

# @retry(wait_fixed=5, stop_max_attempt_number=10)
def get_html():
    html = driver.execute_script("return document.documentElement.outerHTML")
    return html


query_dir = './query_dir'
chrome_options = webdriver.ChromeOptions()

chromedriver = './chromedriver'
os.environ["webdriver.chrome.driver"] = chromedriver

chrome_options.page_load_strategy = 'none'
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('User-Agent=Mozilla/5.0 (Windows NT 6.1; Win64; x64) >AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36 Edg/87.0.664.57')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--hide-scrollbars') #隐藏滚动条, 应对一些特殊页面
chrome_options.add_argument('blink-settings=imagesEnabled=false') #不加载图片, 提升速度
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
#实现规避操作

chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
# chrome_options.add_argument('--ignore-certificate-errors-spki-list')
# chrome_options.add_argument('-ignore-certificate-errors')
# chrome_options.add_argument('-ignore -ssl-errors')
driver = webdriver.Chrome(executable_path=chromedriver,options=chrome_options)
# start_time = datetime.datetime.now()
# web1 = driver.get('https://etherscan.io/tx/{}/')
# driver.close()
# end_time = datetime.datetime.now()
# print(end_time - start_time)
# driver=webdriver.PhantomJS(service_args=['--ssl-protocol=any'])  
# or
# driver = webdriver.PhantomJS( service_args=['--ignore-ssl-errors=true'])



token1155List = pd.read_csv('tokenlist.csv')
tokenlist = token1155List.apply(lambda x: tuple(x), axis=1).values.tolist()


for eachtoken in tokenlist:
    print("eachtoken[0]=",eachtoken[0])
    txs_details = []
    with open("./txs-details/{}.csv".format(eachtoken[0]),"a",newline='', encoding = 'utf-8') as csvfile: 
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["token","txhash","timestamp","from","to","ERCtype","tokenId","transferedTokenAddress","transferedAmount","value","transactionFee","gasPrice",'fullStr'])
        writer = csv.writer(csvfile)
        hashes = pd.read_csv('./txhash-Noduplicates/{}.csv'.format(eachtoken[0]))
        hashList = list(hashes['txhash'])
        # print("hashList=",hashList)
        for hash in hashList:
            print("eachtoken[0]=",eachtoken[0],"hash=",hash)
            results = []
            url="http://cn.etherscan.com/tx/{}".format(hash)
            # print(url)
            driver.get(url)
            time.sleep(5)
            
            while(1):
                try:
                    html = driver.page_source         
                    if ('Timestamp' in html)  and ('Value' in html) and ('Transaction Fee' in html) and ('Gas Price' in html) and ('ERC-' in html):
                        break
                    else:
                        time.sleep(5)
                        driver.get(url)
                        time.sleep(5)
                except:
                    time.sleep(5)
                    # driver.quit()
                    driver.get(url)
                    time.sleep(5)

                # html = driver.execute_script("return document.documentElement.outerHTML")
            

            soup = BeautifulSoup(html, "html.parser")
            #----------------timestamp---------------------            
            timestamp = soup.find('div',id="ContentPlaceHolder1_divTimeStamp").text
            timestamp = timestamp.replace('\\s','')
            timestamp = timestamp.replace('\n','')  

            #-------------------value-----------------------
            value = soup.find('span',id="ContentPlaceHolder1_spanValue").text.replace('\\s','')
            value = value.replace('\n','')
            # print(value)

            #-------------------transactionFee-----------------------
            transactionFee = soup.find('span',id="ContentPlaceHolder1_spanTxFee").text.replace('\\s','')
            transactionFee = transactionFee.replace('\n','')
            # print(transactionFee)

            #-------------------gasPrice-----------------------
            gasPrice = soup.find('span',id="ContentPlaceHolder1_spanGasPrice").text.replace('\\s','')
            gasPrice = gasPrice.replace('\n','')
            # print(gasPrice)

            #------------------find_transactions------------
            txs = soup.find_all('li',class_="d-flex flex-wrap")
            for tx in txs:
                fullStr = str(tx.text).encode(encoding='utf-8')
                fullStr = str(fullStr)
                hrefs = tx.find_all('a', href=re.compile('/token/'))

                #------------------transferedAmount------------
                if '1 of Token ID' in fullStr:
                    transferedAmount = 1
                else:
                    transferedAmount = 'Find in fullStr'

                #------------------------ERCtype-------------------
                if 'ERC-1155' in fullStr:
                    ERCtype = 'ERC-1155'
                elif 'ERC-721' in fullStr:
                    ERCtype = 'ERC-721'
                else:
                    ERCtype = 'Find in fullStr'
                # print(ERCtype)

                fromAddr = hrefs[0].get('href')
                print("type(fromAddr)=",type(fromAddr))
                toAddr = hrefs[1].get('href')
                transferedTokenAddress = hrefs[-1].get('href')
                try:
                    tokenId = hrefs[2].get('href')
                except:
                    tokenId = 'None'

                results.append((eachtoken[0],hash,timestamp,fromAddr,toAddr,ERCtype,tokenId,transferedTokenAddress,transferedAmount,value,transactionFee,gasPrice,fullStr))
                
            writer.writerows(results)
        
    csvfile.close()