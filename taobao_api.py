import requests as reqs
import base64
import logging
import re
import json

# 调用淘宝识图请求接口，获取商品源和标签

with open("./product-recognition/pic/高跟鞋/103798.png", "rb") as f:
    # b64encode：编码，b64decode: 解码
    base64_data = base64.b64encode(f.read())

imagefile={ "file": ('103798.png', open("./product-recognition/pic/高跟鞋/103798.png", "rb"), "image/png")}
header={
'accept': 'application/json, text/javascript, */*; q=0.01',
'accept-encoding': 'gzip, deflate, br',
'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
'cookie': 'miid=1071717533349279183; cna=KhCIE0bWfhwCAX0hn5yH62W+; hng=CN%7Czh-CN%7CCNY%7C156; thw=cn; tracknick=%5Cu6012%5Cu7FFC%5Cu523A%5Cu5929; tg=0; cookie2=37ce11630c799a34db3eeabd249f6a62; v=0; _tb_token_=7573ee0e33915; dnk=%5Cu6012%5Cu7FFC%5Cu523A%5Cu5929; enc=1ZZJ7ftm7ugBQPv7hC%2BdRiJhhmSn94OyhcNu5nXV0ueHyHKwg69Zy6T3uds6lRhIITodY2O%2BTPCjAuvMNx3edg%3D%3D; ali_ab=125.122.209.201.1560240447102.8; alitrackid=www.taobao.com; lastalitrackid=www.taobao.com; t=0a76d28c8b2a7399062d6f57fd1d001e; JSESSIONID=5A7FB3C5F963EC71B2CA3B3E5651D44C; unb=858120373; lgc=%5Cu6012%5Cu7FFC%5Cu523A%5Cu5929; cookie17=W8zD%2FFQJTZJs; _l_g_=Ug%3D%3D; sg=%E5%A4%A938; _nk_=%5Cu6012%5Cu7FFC%5Cu523A%5Cu5929; cookie1=UUAOyZmyJUk%2FnjoLE1Xv0VlpxHN0%2FI%2F9BmBQuzuVDM8%3D; mt=ci=12_1; uc1=pas=0&cookie16=URm48syIJ1yk0MX2J7mAAEhTuw%3D%3D&cookie15=UIHiLt3xD8xYTw%3D%3D&existShop=false&cookie14=UoTbnxEivIIBSA%3D%3D&tag=8&cookie21=U%2BGCWk%2F7pY%2FF&lng=zh_CN; uc3=vt3=F8dByuay3mFM%2BSYeuSI%3D&lg2=V32FPkk%2Fw0dUvg%3D%3D&nk2=pjrdTI2nxtg%3D&id2=W8zD%2FFQJTZJs; csg=8be66c30; skt=59f493a5682404f1; existShop=MTU3MjkyNDgwMQ%3D%3D; uc4=nk4=0%40pPVfia%2FLuQSrLt%2BzQvnRMla1jQ%3D%3D&id4=0%40We8we0iAzFHCIZjZl6YGflCfUCo%3D; _cc_=U%2BGCWk%2F7og%3D%3D; l=dBOE-287vArICSuFBOfaourza77T4IRbzsPzaNbMiICPOufH50lCWZdxAcLMCnGVnsKwR3rxvJx0BrLNbP5-6NEUE3k_J_YK3d8h.; isg=BCYmjyPZR3GBHhXrUfUmgg-3d5por2-nIQei9RDPyckkk8SteJQ10UA168-6O2LZ',
'origin': 'https://s.taobao.com',
'referer': 'https://s.taobao.com/search?q=&imgfile=&js=1&stats_click=search_radio_all%253A1&initiative_id=staobaoz_20191105&ie=utf8&tfsid=O1CN01g6xEgi1TxE8ft6Nmb_!!0-imgsearch.jpg&app=imgsearch',
'sec-fetch-mode': 'cors',
'sec-fetch-site': 'same-origin',
'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.70 Safari/537.36',
'x-requested-with': 'XMLHttpRequest'}

res_pic=reqs.post('https://s.taobao.com/image',files=imagefile,headers=header, verify=False)

print(res_pic.json()['name'])

res_alias=reqs.get("""https://s.taobao.com/search?q=&imgfile=&js=1&stats_click=search_radio_all&initiative_id=staobaoz_20191105&ie=utf8&tfsid=%s&app=imgsearch""" % res_pic.json()['name'],{},headers=header,verify=False)
#reg=r'<script>+[\W]+g_page_config = ([\w\W]+"map":{}};);+[\W]+<\/script>'
reg_alias=r'<script>+[\W]+g_page_config = ([\w\W]+"map":{}})+'

m=re.search(reg_alias,res_alias.text,re.M|re.I)
data=json.loads(m.group(1))

# 外观相似宝贝
detail = data['mods']['itemlist']['data']['collections'][0]['auctions'][0]

# 您可能会喜欢
# item = data['mods']['itemlist']['data']['collections'][1]

#for detail in item['auctions']:
print('商品：',detail['title'],detail['pic_url'],detail['detail_url'])
res_detail=reqs.get("https:"+detail['detail_url'],{},headers=header,verify=False)

reg_detail=r'"attributes-list"+([\w\W]+)</ul>+'

m=re.search(reg_detail,res_detail.text,re.M|re.I)
detail=m.group(1).replace('\t','').replace('"','').replace('\'','').replace(' ','').replace('&nbsp;','').replace('\r','').replace('\n','').replace('<li','')
field_list=detail.split('</li>')

for field in field_list[0:-1]:
    try:
        f_obj=field.split('>')[-1].split(':')
        f_key=f_obj[0]
        f_value=f_obj[1]
        print('属性：',f_key,f_value)
    except Exception as e:
        print(e)
        pass
            

