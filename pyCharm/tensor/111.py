#-*- coding: utf-8 -*-

import urllib.request
from urllib import parse
import json
import sys as sss
import sys
print(sys.stdin.encoding)

kwd = "자바"
documentUrl = 'http://10.100.0.209:6166/search?select=*&from=recruit_search.recruit_search'
url = 'http://10.100.0.209:6166/search?select=*&from=recruit_search.recruit_search&limit=99999&where=' \
      + parse.quote('title_idx=\'' + kwd + '\'')

documentfp = urllib.request.urlopen(documentUrl)
termfp = urllib.request.urlopen(url)


DocumentData = documentfp.read().decode('utf-8')
TermData = termfp.read().decode('utf-8')

##print(TermData)

term_data = json.loads(TermData)
document_data = json.loads(DocumentData)

term_cnt = term_data['result']['total_count']
document_cnt = document_data['result']['total_count']

print(term_cnt / document_cnt * 100)

TermData
documentfp.close()
termfp.close()
