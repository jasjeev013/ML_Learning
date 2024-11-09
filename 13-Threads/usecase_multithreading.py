'''
https://python.langchain.com/docs/tutorials/

https://docs.smith.langchain.com/

https://langchain-ai.github.io/langgraph/
'''

import threading
import requests
from bs4 import BeautifulSoup

urls = ['https://python.langchain.com/docs/tutorials/','https://docs.smith.langchain.com/','https://langchain-ai.github.io/langgraph/']

def fetch_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    print('the length of scraping is : ',len(soup.text), 'for url : ', url)
    print('----------------------------------------')

threads = []

for url in urls:
    thread = threading.Thread(target=fetch_content, args=(url,))
    threads.append(thread)
    thread.start()
    
    
for thread in threads:
    thread.join()
    
print('All threads are done')