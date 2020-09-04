import urllib.request
import re
import time
import socket
import threading
# 设置超时10s
socket.setdefaulttimeout(10)


t1='http://konachan.net/post?page='
t2='&tags='
op=urllib.request.build_opener()
headers = ('User-Agent','Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11')
op.addheaders=[headers]
n=0
global a
def xz(url,name):
    for kkk in range(6):
        try:
            urllib.request.urlretrieve(url,'%s.jpg'%name)
        except Exception as r:
            print(name,'获取错误 %s'%r)
        else:
            print(name,"获取成功")
            break

for vivo in range(1,1001):#要爬取的页数，可以自己设置
    for op in range(10000000):
        try:
            a=urllib.request.urlopen(t1+str(vivo)+t2).read().decode('utf-8')
        except Exception as r:
            print('网页获取失败',r)
            time.sleep(5)
        else:
            print("网页获取成功")
            break

    print(a)
    c=re.findall('href="http[^"]*[jp][pn]g">',a)
    ttt=[]
    for i in c:
        n+=1
        #print(i[6:-2])
        print("add",n)
        t=threading.Thread(target=xz, args=(i[6:-2],n,))
        t.setDaemon(True)
        ttt.append(t)
    for h in ttt:
        h.start()
    for h in ttt:
        h.join()
    print('第',vivo,'页成功')


