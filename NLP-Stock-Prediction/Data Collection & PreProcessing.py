#!/usr/bin/env python
# coding: utf-8

# ### Data Collection
# 
# 1. Define Classes & Functions <br>
#     a. Get list of SEC docs for CIK<br>
#     b. Extract text, date, item numbers from each link<br>
#     c. Get price given ticker, data from AlphaVantage<br>
#     d. Get movement given ticker, date<br>
#     e. Get index movement<br>
#     f. Check if date is a weekday, and if necessary, adjust to Friday before<br>
#     g. Calculate dates for month before, quarter before, year before for historical movement calculations
# 2. GET S&P 500 company info<br>
# 3. Get list of 8K doc links <br>
# 4. Download 8Ks & Stock Movements<br>

# In[1]:


from bs4 import BeautifulSoup
import datetime
import unicodedata
import requests
import pandas as pd
import numpy as np
from time import sleep
import math
from config import Config
import dateutil.relativedelta
import pandas_market_calendars as mcal
import os
import io
import re
from tqdm import tqdm
import gc
import ast


# ### 1 Define Functions and Classes

# In[2]:


from bs4 import BeautifulSoup
import datetime
import unicodedata
import requests
import pandas as pd
import numpy as np
from time import sleep
import re

class SEC_Extractor:
    def get_doc_links(cik,ticker):
        try:
            base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
            inputted_cik = cik
            payload = {
                "action" : "getcompany",
                "CIK" : inputted_cik,
                "type" : "8-K",
                "output":"xml",
                "dateb" : "20180401",
            }
            sec_response = requests.get(url=base_url,params=payload)
            soup = BeautifulSoup(sec_response.text,'lxml')
            url_list = soup.findAll('filinghref')
            html_list = []
            # Get html version of links
            for link in url_list:
                link = link.string
                if link.split(".")[len(link.split("."))-1] == 'htm':
                    txtlink = link + "l"
                    html_list.append(txtlink)

            doc_list = []
            doc_name_list = []
            # Get links for txt versions of files
            for k in range(len(html_list)):
                txt_doc = html_list[k].replace("-index.html",".txt")
                doc_name = txt_doc.split("/")[-1]
                doc_list.append(txt_doc)
                doc_name_list.append(doc_name)
                # Create dataframe of CIK, doc name, and txt link
            df = pd.DataFrame(
                {
                "cik" : [cik]*len(html_list),
                "ticker" : [ticker]*len(html_list),
                "txt_link" : doc_list,
                "doc_name": doc_name_list
                }
            )
        except requests.exceptions.ConnectionError:
                sleep(.1)
        return df

    # Extracts text and submission datetime from document link
    def extract_text(link):
        try:
            r = requests.get(link)
            #Parse 8-K document
            filing = BeautifulSoup(r.content,"html5lib",from_encoding="ascii")
            #Extract datetime
            try:
                submission_dt = filing.find("acceptance-datetime").string[:14]
            except AttributeError:
                    # Flag docs with missing data as May 1 2018 10AM
                submission_dt = "20180501100000"
            
            submission_dt = datetime.datetime.strptime(submission_dt,"%Y%m%d%H%M%S")
            #Extract HTML sections
            for section in filing.findAll("html"):
                try:
                    #Remove tables
                    for table in section("table"):
                        table.decompose()
                    #Convert to unicode
                    section = unicodedata.normalize("NFKD",section.text)
                    section = section.replace("\t"," ").replace("\n"," ").replace("/s"," ").replace("\'","'")            
                except AttributeError:
                    section = str(section.encode('utf-8'))
            filing = "".join((section))
        except requests.exceptions.ConnectionError:
                sleep(10)
        sleep(.1)

        return filing, submission_dt

    def extract_item_no(document):
        pattern = re.compile("Item+ +\d+[\:,\.]+\d+\d")
        item_list = re.findall(pattern,document)
        return item_list


# In[3]:


# Returns Dataframe of document links for a given CIK
idx = pd.Index
class FinDataExtractor:
    def __init__(self):
        # S&P 500 index data downloaded from Yahoo Finance GSPC
        self.gspc_df = pd.read_csv("Data/Indexes/gspc.csv",parse_dates=['Date'],index_col="Date")
        # Get VIX index data downloaded from Yahoo Finance
        self.vix_df = pd.read_csv("Data/Indexes/vix.csv",parse_dates=['Date'],index_col="Date")
        nyse = mcal.get_calendar('NYSE')
        self.nyse_holidays = nyse.holidays().holidays
        self.all_tickers_data = pd.read_pickle("Pickles/all_tickers_data.pkl")
        
#Takes datetime object and ticker string, returns price (opening or closing)
    def get_historical_movements(self,row,period):
        ticker,release_date = row[0],row[1]

       #1 Week
        if period == "week":
            e_start = release_date + datetime.timedelta(weeks=-1)
            b_start = e_start

            e_end = release_date + dateutil.relativedelta.relativedelta(days=-1)
            b_end = e_end

         #1 Month    
        elif period == "month":
            e_start = release_date + dateutil.relativedelta.relativedelta(months=-1)
            b_start = e_start + dateutil.relativedelta.relativedelta(days=-5)

            e_end = release_date + dateutil.relativedelta.relativedelta(days=-1)
            b_end = release_date + dateutil.relativedelta.relativedelta(days=-6)

        #1 Quarter
        elif period == "quarter":
            e_start = release_date + dateutil.relativedelta.relativedelta(months=-3)
            b_start = e_start + dateutil.relativedelta.relativedelta(days=-10)

            e_end = release_date + dateutil.relativedelta.relativedelta(days=-1)
            b_end = release_date + dateutil.relativedelta.relativedelta(days=-11)

        #1 Year
        elif period == "year":
            e_start = release_date + dateutil.relativedelta.relativedelta(years=-1)
            b_start = e_start + dateutil.relativedelta.relativedelta(days=-20)

            e_end = release_date + dateutil.relativedelta.relativedelta(days=-1)
            b_end = release_date + dateutil.relativedelta.relativedelta(days=-21)
        else:
            raise KeyError

        e_start = self.weekday_check(e_start)
        b_start = self.weekday_check(b_start)
        e_end = self.weekday_check(e_end)
        b_end = self.weekday_check(b_end)

        start_price = self.get_av_data(ticker=ticker,start_date = b_start, end_date = e_start)
        end_price = self.get_av_data(ticker=ticker,start_date = b_end, end_date = e_end)
        stock_change = self.calculate_pct_change(end_price,start_price)

        start_index = self.get_index_price(start_date = b_start, end_date = e_start)
        end_index = self.get_index_price(start_date = e_start, end_date = e_end)
        index_change =  self.calculate_pct_change(end_index,start_index)

        normalized = stock_change - index_change
        return normalized

    def get_av_data(self,ticker,start_date,end_date,market_open=False):
        start_date = start_date.date()
        end_date = end_date.date()

        try:
            if market_open == False:
                price = self.all_tickers_data.xs(ticker,0).loc[end_date:start_date,"adjusted_close"].mean()
            else:
                price = self.all_tickers_data.xs(ticker,0).loc[end_date:start_date,"open"].mean()
        except (KeyError,IndexError):
            price = np.nan
        return price

    # Takes ticker, 8K release date, checks time of release and then calculate before and after price change
    def get_change(self,row):
        release_date = row['release_date']
        ticker = row['ticker']
        market_close = release_date.replace(hour=16,minute=0,second=0)
        market_open = release_date.replace(hour=9,minute=30,second=0)

    # If report is released after market hours, take change of start date close and release date open
        if release_date > market_close:
            start_date = release_date
            end_date = release_date + datetime.timedelta(days=1)
            end_date = self.weekday_check(end_date)

            price_before_release = self.get_av_data(ticker,start_date,start_date,market_open=False)
            price_after_release = self.get_av_data(ticker,end_date,end_date,market_open=True)

            index_before_release = self.get_index_price(start_date,start_date,market_open=False)
            index_after_release = self.get_index_price(end_date,end_date,market_open=True)

            try:
                vix = self.vix_df.loc[self.vix_df.index == np.datetime64(start_date.date()),"Adj Close"][0].item()
            except IndexError:
                vix = np.nan

        # If report is released before market hours, take change of start date's close and release date's open
        elif release_date < market_open:
            start_date = release_date + datetime.timedelta(days=-1)
            start_date = self.weekday_check(start_date)
            end_date = release_date

            price_before_release = self.get_av_data(ticker,start_date,start_date,market_open=False)
            price_after_release = self.get_av_data(ticker,end_date,end_date,market_open=True) 

            index_before_release = self.get_index_price(start_date,start_date,market_open=False)
            index_after_release = self.get_index_price(end_date,end_date,market_open=True)
            try:
                vix = self.vix_df.loc[self.vix_df.index == np.datetime64(start_date.date()),"Adj Close"][0].item()
            except IndexError:
                vix = np.nan
        # If report is released during market hours, use market close
        else:
            start_date = release_date
            end_date = release_date
            price_before_release = self.get_av_data(ticker,start_date,start_date,market_open=True)
            price_after_release = self.get_av_data(ticker,end_date,end_date,market_open=False)

            index_before_release = self.get_index_price(start_date,start_date,market_open=True)
            index_after_release = self.get_index_price(end_date,end_date,market_open=False)
            
            try:
                vix = self.vix_df.loc[self.vix_df.index == np.datetime64(start_date.date()),"Open"][0].item()
            except IndexError:
                vix = np.nan
                
        price_pct_change = self.calculate_pct_change(price_after_release,price_before_release)
        index_pct_change = self.calculate_pct_change(index_after_release,index_before_release)
        normalized_change = price_pct_change - index_pct_change

        return normalized_change, vix

    def get_index_price(self,start_date,end_date,market_open=False):
        try:
            if market_open == True:
                price = self.gspc_df.loc[(self.gspc_df.index >= np.datetime64(start_date.date())) & 
                                 (self.gspc_df.index <= np.datetime64(end_date)),"Open"].mean()
            else:
                price = self.gspc_df.loc[(self.gspc_df.index >= np.datetime64(start_date.date())) & 
                                 (self.gspc_df.index <= np.datetime64(end_date)),"Adj Close"].mean()
        except IndexError:
                price = np.nan
        return price

    def calculate_pct_change(self,end_value,start_value):
        pct_change = (end_value - start_value) / start_value
        pct_change = round(pct_change,4) * 100
        return pct_change

    def weekday_check(self,date):  
        while date.isoweekday() > 5 or date.date() in self.nyse_holidays:
            date = date + datetime.timedelta(days=-1)
        return date


# ### 2. Get S&P 500 Data

# In[4]:


# Get table of the S&P 500 tickers, CIK, and industry from Wikipedia
wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
cik_df = pd.read_html(wiki_url,header=0,index_col=0)[0]
cik_df['GICS Sector'] = cik_df['GICS Sector'].astype("category")
cik_df['GICS Sub Industry'] = cik_df['GICS Sector'].astype("category")
cik_df.head()


# ### 3. Get List of 8K links from SEC Edgar

# In[5]:


sec_ext = SEC_Extractor
no_parts = 2
part_no = 3


# In[ ]:


df_list = []
company_list = cik_df['CIK'].to_dict()
for (ticker,cik) in tqdm(company_list.items()):
    df_list.append(sec_ext.get_doc_links(cik,ticker))
doc_links_df = pd.concat(df_list,axis=0)
doc_links_df = doc_links_df.set_index("ticker").join(cik_df['GICS Sector']).join(cik_df['GICS Sub Industry']).reset_index().rename(columns={"index":"ticker"})
doc_links_df.head()


# In[ ]:


doc_links_df.to_pickle("Pickles/doc_links_df.pkl")


# ### 4. Download 8Ks & Stock Movements

# In[6]:


while part_no > no_parts:
    no_parts = int(input("Split data into how many parts?"))
    part_no = int(input("Which part is this?"))
    
chunksize = int(input("Number of rows to process at once (10 to 50 recommended)"))

#Load pickle
crawled_df = np.array_split(pd.read_pickle("Pickles/doc_links_df.pkl"),no_parts)[part_no-1][:10]
crawled_len = len(crawled_df['txt_link'])
chunks = math.ceil(crawled_len/chunksize)

df_list = []
for i, df in tqdm(enumerate(np.array_split(crawled_df,chunks))):
    df['text'], df['release_date'] = zip(*df['txt_link'].apply(sec_ext.extract_text))
    df['items'] = df['text'].map(sec_ext.extract_item_no)
    if not os.path.isfile("Data/texts_example{}.csv.gzip".format(part_no)): #If no file exists, create one with header
        df.to_csv("Data/texts_example{}.csv.gzip".format(part_no),chunksize=chunksize,compression="gzip")
    else: # else it exists so append without writing the header
        df.to_csv("Data/texts_example{}.csv.gzip".format(part_no),mode="a",header=False,compression="gzip",chunksize=chunksize)       
    df_list.append(df)
    del df
    
    if i % 50 == 0:
        gc.collect()
df = pd.concat(df_list)
df.head()


# ### Load Financial Data

# In[7]:


cik_dict = cik_df['CIK'].to_dict()
cik_dict = {v: k for k, v in cik_dict.items()}
df['ticker'] = df['cik'].map(cik_dict)


# In[9]:


df1_gen = pd.read_csv("Data/texts1.csv.gzip",compression="gzip",parse_dates=['release_date'],chunksize=1000,index_col=[0])
#df2_gen = pd.read_csv("Data/texts2.csv.gzip",compression="gzip",parse_dates=['release_date'],chunksize=1000)
df1 = pd.concat([df for df in df1_gen])
#df2 = pd.concat([df for df in df2_gen])
#df2 = pd.read_csv("Data/texts2.csv",parse_dates=['release_date'],encoding="utf_8",index_col=[0])
gc.collect()
df = pd.concat([df1,df2],axis=0)
gc.collect()
df['items'] = df['items'].map(lambda x: ast.literal_eval(x))


# In[ ]:


df.head()


# In[ ]:


# Find rows flagged where no date was found
df = df.loc[~(df['release_date'] >= pd.datetime(year=2018,month=5,day=1))]
df = df.drop_duplicates(subset="doc_name")
df.index.names = ['ticker']
df = df.reset_index()


# In[ ]:


gc.collect()


# In[ ]:


#from FinDataExtractor import FinDataExtractor
fin_data = FinDataExtractor()
## Load pickle of ticker, date, and doc number

df['price_change'],df['vix'] = zip(*df[['ticker','release_date']].apply(fin_data.get_change,axis=1))
df['rm_week'] = df[['ticker','release_date']].apply(fin_data.get_historical_movements,period="week",axis=1)
df['rm_month'] = df[['ticker','release_date']].apply(fin_data.get_historical_movements,period="month",axis=1)
df['rm_qtr'] = df[['ticker','release_date']].apply(fin_data.get_historical_movements,period="quarter",axis=1)
df['rm_year'] = df[['ticker','release_date']].apply(fin_data.get_historical_movements,period="year",axis=1)
df["signal"] = df['price_change'].map(lambda x: "stay" if -1<x<1 else ("up" if x>1 else "down"))
df.head()


# In[ ]:


chunks = 20
for i, df_part in tqdm(enumerate(np.array_split(df,chunks))):
    if not os.path.isfile("Data/texts_and_fin.csv"): #If no file exists, create one with header
        df_part.to_csv("Data/texts_and_fin.csv")
    else: # else it exists so append without writing the header
        df_part.to_csv("Data/texts_and_fin.csv",mode="a",header=False,)       
    del df_part
    
    if i % 50 == 0:
        gc.collect()


# In[ ]:





# In[ ]:




