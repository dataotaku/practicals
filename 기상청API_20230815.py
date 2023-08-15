import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

# 필요한 패키지 설치
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import os
from dateutil.relativedelta import relativedelta

import requests
import os
import shutil

import time
import math
from datetime import datetime
from datetime import timedelta
from dateutil import relativedelta
from itertools import compress
import psycopg2 as psql

import time
import requests
# import xml.etree.ElementTree as ET
# from bs4 import BeautifulSoup

from xml_to_dict import XMLtoDict
import traceback
import pickle


# 일반인증키 (디코딩)
key1 = 'FwXSGyC7A5FCoJ8kPJItiFvQ41k2jW9kxOW%2BxUSJ%2BEfwsQa4BkptDnUXwp3l%2BE8qEPQ4DmDHPS3KzkbBonWI4g%3D%3D'
# (인코딩)
key2 = 'FwXSGyC7A5FCoJ8kPJItiFvQ41k2jW9kxOW+xUSJ+EfwsQa4BkptDnUXwp3l+E8qEPQ4DmDHPS3KzkbBonWI4g=='

code=pd.read_excel("./data/지역별 지점코드(20220701).xlsx")
code.dtypes
code['1단계'].head()
code.loc[pd.Series.isnull(code['2단계']), '행정구역코드'].astype(str)
code[~pd.Series.isnull(code['2단계']) & pd.Series.isnull(code['3단계'])].shape
code.head()



def get_level_one():
    """
    시구 기준의 지역코드를 불러옵니다.
    """
    code=pd.read_excel("./data/지역별 지점코드(20220701).xlsx")
    # code=pd.read_excel("./data/KIKcd_B_20180122.xlsx")
    # code["ji_code"] = code["법정동코드"].astype(str).str[0:5]
    code_sido = code.loc[pd.Series.isnull(code['2단계']), '행정구역코드'].astype(str)
    # code_sigu.columns = ["ji_code", "si", "gu"]
    # code_sigu = code_sigu.drop_duplicates()
    # code_sigu = code_sigu.reset_index(drop=True)
    # code_sigu = code_sigu.fillna("-")

    return code_sido

len(get_level_one())

def get_level_two():
    """
    시구 기준의 지역코드를 불러옵니다.
    """
    code=pd.read_excel("./data/지역별 지점코드(20220701).xlsx")
    # code=pd.read_excel("./data/KIKcd_B_20180122.xlsx")
    # code["ji_code"] = code["법정동코드"].astype(str).str[0:5]
    code_sgg = code.loc[(code['1단계'] == '경기도') & (code['2단계'].isin(['의정부시', '동두천시', '파주시',\
                                                                    ]) != True) & pd.Series.isnull(code['3단계']), '행정구역코드'].astype(str)
    # code_sigu.columns = ["ji_code", "si", "gu"]
    # code_sigu = code_sigu.drop_duplicates()
    # code_sigu = code_sigu.reset_index(drop=True)
    # code_sigu = code_sigu.fillna("-")

    return code_sgg

len(get_level_two())




def remove_for_price(val):
    val = val.replace(" ","")
    val = val.replace(",", "")

    return val


start_str = (datetime.today() + timedelta(days=-1)).strftime("%Y%m%d")
start_date = datetime.strptime(start_str, "%Y%m%d")
start_date
numdays = 1
day_here = []
for x in range(0, numdays):
    day_date = start_date + timedelta(days=x)
    day_here.append(day_date.strftime("%Y%m%d"))
print(day_here)
hour_list = [f'{x:02d}' for x in list(range(24))]

time_here = []
for x in day_here:
    for y in hour_list:
        time_here.append(x+y)

len(time_here)






API_KEY = key1
diffusion_url = 'http://apis.data.go.kr/1360000/LivingWthrIdxServiceV4/getAirDiffusionIdxV4'
# queryParams = '?' + urlencode({ quote_plus('ServiceKey') : '서비스키', quote_plus('serviceKey') : '-', quote_plus('areaNo') : '1100000000', quote_plus('time') : '2021070618', quote_plus('dataType') : 'xml' })

area_no = get_level_one()

url_list = []

for l in area_no:
    for time_tgt in time_here:
        url_list.append("{}?ServiceKey={}&areaNo={}&time={}&dataType={}".format(diffusion_url, API_KEY, l, time_tgt,'xml'))

sido_urls = url_list

print(sido_urls[::3])
len(sido_urls[::3])


area_no = get_level_two()
len(area_no)

url_list = []

for l in area_no:
    for time_tgt in time_here:
        url_list.append("{}?ServiceKey={}&areaNo={}&time={}&dataType={}".format(diffusion_url, API_KEY, l, time_tgt,'xml'))
        # url_list.append("{}?LAWD_CD={}&DEAL_YMD={}&serviceKey={}".format(biz_url, l, ym_tgt, API_KEY))
        # url_list.append("{}?LAWD_CD={}&DEAL_YMD={}&serviceKey={}".format(dandok_url, l, ym_tgt, API_KEY))

sgg_urls = url_list

# print(sgg_urls)
len(sgg_urls)

yesterday_urls = sido_urls[::3] + sgg_urls[::3]
len(yesterday_urls)

datum = requests.get(yesterday_urls[1])
xd = XMLtoDict()
data = xd.parse(datum.content)
print(data)
dum01 = list(data['response']['body']['items']['item'].keys())[3:]
dum02 = list(data['response']['body']['items']['item'].values())[3:]
dum02 = [x if x != None else np.nan for x in dum02]

dum03 = pd.DataFrame({'time':[int(x.replace('h','')) for x in dum01], 'stagnant_idx':dum02})
dum03['code'] = data['response']['body']['items']['item']['code']
dum03['areaNo'] = data['response']['body']['items']['item']['areaNo']
dum03['date'] = data['response']['body']['items']['item']['date']
dum03


yesterday_aggr = []
for url in yesterday_urls:
    try:
        datum = requests.get(url)

        xd = XMLtoDict()
        data = xd.parse(datum.content)

        # print(data)

        dum01 = list(data['response']['body']['items']['item'].keys())[3:]
        dum02 = list(data['response']['body']['items']['item'].values())[3:]
        dum02 = [x if x != None else np.nan for x in dum02]
        dum03 = pd.DataFrame({'time':[int(x.replace('h','')) for x in dum01], 'stagnant_idx':dum02})
        dum03['code'] = data['response']['body']['items']['item']['code']
        dum03['areaNo'] = data['response']['body']['items']['item']['areaNo']
        dum03['date'] = data['response']['body']['items']['item']['date']
        dum03
        yesterday_aggr.append(dum03)
    except:
        pass

yesterday_aggr = pd.concat(yesterday_aggr)

yesterday_aggr['stagnant_idx'] = yesterday_aggr['stagnant_idx'].astype(str)
yesterday_aggr['time'] = yesterday_aggr['time'].astype(str)
yesterday_aggr.shape
diffu_data = yesterday_aggr.drop_duplicates().copy()
diffu_data.shape

with open("D:/pm_data/air_diffusion/diffu_"+ start_str + '.pkl', 'wb') as dif:
    pickle.dump(diffu_data, dif)



host = "localhost"
port = "5432"
user = "dataotaku"
password = "!@#qwe5392"
db = "particle"

conn = psql.connect(host=host, port=port, user=user, password=password, dbname=db)
cur = conn.cursor()
cur
# cur.execute("select * from tb_sal_his; ")
# cur.fetchall()

cur.execute("""create table if not exists air_diffusion (time varchar(2), stagnant_idx varchar(3), 
code varchar(3), areaNo varchar(10), date varchar(10), primary key(date, areaNo, time));""")

for k in range(diffu_data.shape[0]):
    try:
        cur.execute("""insert into air_diffusion
                        ( time, stagnant_idx, code, areaNo, date) VALUES (%s, %s, %s, %s, %s)
        """, (tuple(diffu_data.iloc[k,:]))
        )
        conn.commit()
    except:
        pass


cur.close()
conn.close()



import time
import requests
# import xml.etree.ElementTree as ET
# from bs4 import BeautifulSoup

from xml_to_dict import XMLtoDict
import pickle

def ldaps2xy(X, Y, x0=83.64379, y0=412.2285):
    x1 = X-1
    y1 = Y-1 
    NX = 426 # l 426
    NY = 498 # l 498
    map_Re=6371.00877
    map_grid=1.5
    map_slat1=30.0
    map_slat2=60.0
    map_olon=126.0
    map_olat=38.0
    map_x0= x0 -1 #83.64379
    map_y0= y0 -1 #412.2285
    
    PI=math.pi
    DEGRAD=PI/180.0
    RADDEG=180.0/PI
    
    re=map_Re/map_grid
    slat1=map_slat1*DEGRAD
    slat2=map_slat2*DEGRAD
    olon=map_olon*DEGRAD
    olat=map_olat*DEGRAD
    
    sn = math.tan(PI*0.25 + slat2*0.5) / math.tan(PI*0.25 + slat1*0.5)
    sn = math.log(math.cos(slat1)/math.cos(slat2))/math.log(sn)
    sf = math.tan(PI*0.25 + slat1*0.5)
    sf = ((sf**sn)*math.cos(slat1))/sn
    ro = math.tan(PI*0.25 + olat*0.5)
    ro = re*sf/(ro**sn)
    
    xn = x1 - map_x0
    yn = ro - y1 + map_y0

    # 여기서 갈림
    ra = math.sqrt(xn*xn + yn*yn)
    if (sn < 0):
        ra = -ra
    
    alat = ((re*sf)/ra)**(1.0/sn)
    alat = 2.0*math.atan(alat) - PI*0.5
    
    if (abs(xn) <= 0.0):
        theta = 0.0
    else:
        if (abs(yn) <= 0.0):
            theta = PI*0.5
            if (xn < 0.0):
                theta = -theta
        else:
            theta = math.atan2(xn, yn)
    alon = (theta / sn) + olon
    lat = alat * RADDEG
    lon = alon * RADDEG
    # res = [lon, lat]

    return lon, lat

ldaps2xy(426,498) #126.929810, 37.488201


def xy2ldaps(lon, lat, x0=83.64379, y0=412.2285):
    NX = 426 # l 426
    NY = 498 # l 498
    map_Re=6371.00877
    map_grid=1.5
    map_slat1=30.0
    map_slat2=60.0
    map_olon=126.0
    map_olat=38.0
    map_x0= x0 -1#83.64379
    map_y0= y0 -1#412.2285
    
    PI=math.pi
    DEGRAD=PI/180.0
    RADDEG=180.0/PI
    
    re=map_Re/map_grid
    slat1=map_slat1*DEGRAD
    slat2=map_slat2*DEGRAD
    olon=map_olon*DEGRAD
    olat=map_olat*DEGRAD
    
    sn = math.tan(PI*0.25 + slat2*0.5) / math.tan(PI*0.25 + slat1*0.5)
    sn = math.log(math.cos(slat1)/math.cos(slat2))/math.log(sn)
    sf = math.tan(PI*0.25 + slat1*0.5)
    sf = (sf**sn)*math.cos(slat1)/sn
    ro = math.tan(PI*0.25 + olat*0.5)
    ro = re*sf/(ro**sn)


    # 여기서 갈림
    ra = math.tan(PI*0.25 + lat*DEGRAD*0.5)
    ra = re*sf/(ra**sn)  
    theta = lon * DEGRAD - olon
    
    if (theta > PI):
        theta -= 2.0*PI

    if (theta < -PI):
        theta += 2.0*PI

    theta *= sn


    xn = ra * math.sin(theta) + map_x0
    yn = ro - ra*math.cos(theta) + map_y0

    xn = int(xn + 1.5)
    yn = int(yn + 1.5)

    res = [xn, yn]

    return res

xy2ldaps(126.929810, 37.488201) # 137, 376


def rdaps2xy(X, Y, x0=12.136152, y0=57.11616):
    x1 = X-1
    y1 = Y-1 
    NX = 55 # l 426
    NY = 68 # l 498
    map_Re=6371.00877
    map_grid=12
    map_slat1=30.0
    map_slat2=60.0
    map_olon=126.0
    map_olat=38.0
    map_x0= x0 -1 # 12.136152
    map_y0= y0 -1 # 57.11616
    
    PI=math.pi
    DEGRAD=PI/180.0
    RADDEG=180.0/PI
    
    re=map_Re/map_grid
    slat1=map_slat1*DEGRAD
    slat2=map_slat2*DEGRAD
    olon=map_olon*DEGRAD
    olat=map_olat*DEGRAD
    
    sn = math.tan(PI*0.25 + slat2*0.5) / math.tan(PI*0.25 + slat1*0.5)
    sn = math.log(math.cos(slat1)/math.cos(slat2))/math.log(sn)
    sf = math.tan(PI*0.25 + slat1*0.5)
    sf = ((sf**sn)*math.cos(slat1))/sn
    ro = math.tan(PI*0.25 + olat*0.5)
    ro = re*sf/(ro**sn)
    
    xn = x1 - map_x0
    yn = ro - y1 + map_y0

    # 여기서 갈림
    ra = math.sqrt(xn*xn + yn*yn)
  
    if (sn < 0):
        ra = -ra
    
    alat = ((re*sf)/ra)**(1.0/sn)
    alat = 2.0*math.atan(alat) - PI*0.5
    
    if (abs(xn) <= 0.0):
        theta = 0.0
    else:
        if (abs(yn) <= 0.0):
            theta = PI*0.5
            if (xn < 0.0):
                theta = -theta
        else:
            theta = math.atan2(xn, yn)
    alon = (theta / sn) + olon
    lat = alat * RADDEG
    lon = alon * RADDEG
    res = [lon, lat]

    return res

rdaps2xy(54,52) #131.8648471, 37.2414386


def xy2rdaps(lon, lat, x0=12.136152, y0=57.11616):
    NX = 55 # l 426
    NY = 68 # l 498
    map_Re=6371.00877
    map_grid=12
    map_slat1=30.0
    map_slat2=60.0
    map_olon=126.0
    map_olat=38.0
    map_x0= x0 - 1 # 12.136152
    map_y0= y0 - 1 # 57.11616
    
    PI=math.pi
    DEGRAD=PI/180.0
    RADDEG=180.0/PI
    
    re=map_Re/map_grid
    slat1=map_slat1*DEGRAD
    slat2=map_slat2*DEGRAD
    olon=map_olon*DEGRAD
    olat=map_olat*DEGRAD
    
    sn = math.tan(PI*0.25 + slat2*0.5) / math.tan(PI*0.25 + slat1*0.5)
    sn = math.log(math.cos(slat1)/math.cos(slat2))/math.log(sn)
    sf = math.tan(PI*0.25 + slat1*0.5)
    sf = (sf**sn)*math.cos(slat1)/sn
    ro = math.tan(PI*0.25 + olat*0.5)
    ro = re*sf/(ro**sn)


    # 여기서 갈림
    ra = math.tan(PI*0.25 + lat*DEGRAD*0.5)
    ra = re*sf/(ra**sn)  
    theta = lon * DEGRAD - olon
    
    if (theta > PI):
        theta -= 2.0*PI

    if (theta < -PI):
        theta += 2.0*PI

    theta *= sn


    xn = ra * math.sin(theta) + map_x0
    yn = ro - ra*math.cos(theta) + map_y0

    xn = int(xn + 1.5)
    yn = int(yn + 1.5)

    res = [xn, yn]

    return res

xy2rdaps(131.8648471, 37.2414386) # 126.625369, 37.657464

x_rdaps = list(range(55))
x_rdaps
y_rdaps = list(range(68))
y_rdaps
xy_rdaps = []
for y_ in y_rdaps:
    for x_ in x_rdaps:
        xy_rdaps.append((x_+1, y_+1))
len(xy_rdaps)

x_ldaps = list(range(426))
x_ldaps
y_ldaps = list(range(498))
y_ldaps
xy_ldaps = []
for y_ in y_ldaps:
    for x_ in x_ldaps:
        xy_ldaps.append((x_+1, y_+1))
len(xy_ldaps)


# 일반인증키 (디코딩)
key1 = 'FwXSGyC7A5FCoJ8kPJItiFvQ41k2jW9kxOW%2BxUSJ%2BEfwsQa4BkptDnUXwp3l%2BE8qEPQ4DmDHPS3KzkbBonWI4g%3D%3D'
# (인코딩)
key2 = 'FwXSGyC7A5FCoJ8kPJItiFvQ41k2jW9kxOW+xUSJ+EfwsQa4BkptDnUXwp3l+E8qEPQ4DmDHPS3KzkbBonWI4g=='

# (datetime.today() + timedelta(days=-1)).strftime("%Y%m%d")
start_str = (datetime.today() + timedelta(days=-1)).strftime("%Y%m%d")
start_date = datetime.strptime(start_str, "%Y%m%d")
start_date
numdays = 1
day_here = []
for x in range(0, numdays):
    day_date = start_date + timedelta(days=x)
    day_here.append(day_date.strftime("%Y%m%d"))
print(day_here)
#hour_list = [f'{x:02d}'+'00' for x in list(range(24))]
open_hour = ['0300','0900','1500','2100']

time_here = []
for x in day_here:
    for y in open_hour:
        time_here.append(x+y)

print(time_here)

API_KEY = key1
ldaps1_url = 'http://apis.data.go.kr/1360000/NwpModelInfoService/getLdapsUnisAll'

n=0
ldaps_aggr = []
ldaps_meta_aggr = []
for l in time_here:
    for time_tgt in range(0,49):
        for type in ['Temp','Humi','Wspd','Wdir','Rain']:
            url_list = "{}?ServiceKey={}&baseTime={}&leadHour={}&dataType={}&dataTypeCd={}".format(ldaps1_url, API_KEY, l, time_tgt,'XML', type)
            try:
                n= n+1
                datum = requests.get(url_list)

                xd = XMLtoDict()
                data = xd.parse(datum.content)

                base_time = data['response']['body']['items']['item']['baseTime']
                fcst_time = data['response']['body']['items']['item']['fcstTime']
                grid_km = data['response']['body']['items']['item']['gridKm']
                x_dim = data['response']['body']['items']['item']['xdim']
                y_dim = data['response']['body']['items']['item']['ydim']
                x_origin = data['response']['body']['items']['item']['x0']
                y_origin = data['response']['body']['items']['item']['y0']
                unit_str = data['response']['body']['items']['item']['unit']
                # print(unit_str)
                ldaps_meta = pd.DataFrame({'base':[base_time], 'fcst':[fcst_time], 'type':[type],'grid_km':[grid_km], 'x_dim':[x_dim],
                                           'x_origin':[x_origin], 'y_origin':[y_origin]})

                value_txt = data['response']['body']['items']['item']['value']
                value_num = [float(x) for x in value_txt.split(',')]
                if unit_str == 'mm':
                    df = pd.DataFrame({'idx_xy':xy_ldaps, 'rain':value_num})
                    # print(df.head())
                    # print(df.tail())
                elif unit_str == 'm/s':
                    df = pd.DataFrame({'idx_xy':xy_ldaps, 'wspd':value_num})
                    # print(df.head())
                    # print(df.tail())
                elif unit_str == 'percent':
                    df = pd.DataFrame({'idx_xy':xy_ldaps, 'humi':value_num})
                    # print(df.head())
                    # print(df.tail())
                elif unit_str == 'deg':
                    df = pd.DataFrame({'idx_xy':xy_ldaps, 'wdir':value_num})
                    # print(df.head())
                    # print(df.tail())
                elif unit_str == 'C':
                    df = pd.DataFrame({'idx_xy':xy_ldaps, 'temp':value_num})
                    # print(df.head())
                    # print(df.tail())
                df['base'] = base_time
                df['fcst'] = fcst_time

                if type == 'Temp':
                    print(str(l)+"_"+str(time_tgt)+'_'+str(type))
                    data_df = df
                    print(data_df.shape)
                else:
                    print(str(l)+"_"+str(time_tgt)+'_'+str(type))
                    data_df = data_df.merge(df, on=['base','fcst','idx_xy'], how='left')
                    print(data_df.shape)

                ldaps_meta_aggr.append(ldaps_meta)
            except:
                pass
    ldaps_aggr.append(data_df)

ldaps_aggr
ldaps_aggr = pd.concat(ldaps_aggr)
ldaps_meta_aggr = pd.concat(ldaps_meta_aggr)
ldaps_aggr.tail()
ldaps_aggr.tail()
ldaps_aggr.shape
ldaps_aggr.columns

ldaps_aggr['idx_x'] = ldaps_aggr['idx_xy'].apply(lambda x: str(x[0]))
ldaps_aggr['idx_y'] = ldaps_aggr['idx_xy'].apply(lambda x: str(x[1]))
ldaps_aggr['base'] = ldaps_aggr['base'].astype(str)
ldaps_aggr['fcst'] = ldaps_aggr['fcst'].astype(str)
ldaps_aggr['wdir'] = ldaps_aggr['wdir'].astype(str)
ldaps_aggr['rain'] = ldaps_aggr['rain'].astype(str)
ldaps_aggr['humi'] = ldaps_aggr['humi'].astype(str)
ldaps_aggr['wspd'] = ldaps_aggr['wspd'].astype(str)
ldaps_aggr['temp'] = ldaps_aggr['temp'].astype(str)
ldaps_aggr.drop('idx_xy', axis=1, inplace=True)
ldaps_data = ldaps_aggr.drop_duplicates().copy()
ldaps_aggr = ldaps_data.loc[:,['idx_x','idx_y', 'base','fcst','temp','humi','wspd','wdir','rain']]
ldaps_aggr.head()

res = ldaps_aggr.apply(lambda x: ldaps2xy(np.int64(x[0]), np.int64(x[1])), axis=1)
ldaps_aggr['lon'] = res.apply(lambda x: str(x[0]))
ldaps_aggr['lat'] = res.apply(lambda x: str(x[1]))
ldaps_aggr.head()


with open("D:/pm_data/ldaps_data/ldaps_"+ start_str + '.pkl', 'wb') as dif:
    pickle.dump(ldaps_aggr, dif)

with open("D:/pm_data/ldaps_data/ldaps_meta_"+ start_str + '.pkl', 'wb') as dif:
    pickle.dump(ldaps_meta_aggr, dif)


host = "localhost"
port = "5432"
user = "dataotaku"
password = "!@#qwe5392"
db = "particle"

conn = psql.connect(host=host, port=port, user=user, password=password, dbname=db)
cur = conn.cursor()
cur
# cur.execute("select * from tb_sal_his; ")
# cur.fetchall()

cur.execute("""create table if not exists ldaps_data (idx_x varchar(10), idx_y varchar(10), 
base varchar(20), fcst varchar(20), temp varchar(20), humi varchar(20), wspd varchar(20), wdir varchar(20), rain varchar(20),
lon varchar(20), lat varchar(20),
primary key(idx_x, idx_y, base, fcst));""")

# conn.commit()

for k in range(ldaps_aggr.shape[0]):
    try:
        cur.execute("""insert into ldaps_data
                        (idx_x, idx_y, base, fcst, temp, humi, wspd, wdir, rain, lon, lat) VALUES 
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (tuple(ldaps_aggr.iloc[k,:]))
        )
        conn.commit()
    except:
        pass


cur.close()
conn.close()

# print(n)


# (datetime.today() + timedelta(days=-1)).strftime("%Y%m%d")
start_str = (datetime.today() + timedelta(days=-1)).strftime("%Y%m%d")
start_date = datetime.strptime(start_str, "%Y%m%d")
start_date
numdays = 1
day_here = []
for x in range(0, numdays):
    day_date = start_date + timedelta(days=x)
    day_here.append(day_date.strftime("%Y%m%d"))
print(day_here)
#hour_list = [f'{x:02d}'+'00' for x in list(range(24))]
open_hour = ['0300','0900','1500','2100']

time_here = []
for x in day_here:
    for y in open_hour:
        time_here.append(x+y)

print(time_here)

API_KEY = key1
rdaps1_url = 'http://apis.data.go.kr/1360000/NwpModelInfoService/getRdapsUnisAll'

# 1개 테스트
# rdaps_aggr = []
# for l in time_here:
#     for time_tgt in range(0, 6, 3):
#         for type in ['Temp','Humi','Wspd','Wdir','Rain']:
#             url_list = "{}?ServiceKey={}&baseTime={}&leadHour={}&dataType={}&dataTypeCd={}".format(rdaps1_url, API_KEY, l, time_tgt,'XML', type)
#             datum = requests.get(url_list)
#             xd = XMLtoDict()
#             data = xd.parse(datum.content)
#             print(data)


# 전체 루핑
rdaps_aggr = []
rdaps_meta_aggr = []
for l in time_here:
    for time_tgt in range(0, 90, 3):
        for type in ['Temp','Humi','Wspd','Wdir','Rain']:
            url_list = "{}?ServiceKey={}&baseTime={}&leadHour={}&dataType={}&dataTypeCd={}".format(rdaps1_url, API_KEY, l, time_tgt,'XML', type)
            try:
                datum = requests.get(url_list)

                xd = XMLtoDict()
                data = xd.parse(datum.content)

                base_time = data['response']['body']['items']['item']['baseTime']
                fcst_time = data['response']['body']['items']['item']['fcstTime']
                grid_km = data['response']['body']['items']['item']['gridKm']
                x_dim = data['response']['body']['items']['item']['xdim']
                y_dim = data['response']['body']['items']['item']['ydim']
                x_origin = data['response']['body']['items']['item']['x0']
                y_origin = data['response']['body']['items']['item']['y0']
                unit_str = data['response']['body']['items']['item']['unit']
                # print(unit_str)

                rdaps_meta = pd.DataFrame({'base':[base_time], 'fcst':[fcst_time], 'type':[type],'grid_km':[grid_km], 'x_dim':[x_dim],
                                           'x_origin':[x_origin], 'y_origin':[y_origin]})

                value_txt = data['response']['body']['items']['item']['value']
                value_num = [float(x) for x in value_txt.split(',')]
                if unit_str == 'mm':
                    df = pd.DataFrame({'idx_xy':xy_rdaps, 'rain':value_num})
                    # print(df.head())
                    # print(df.tail())
                elif unit_str == 'm/s':
                    df = pd.DataFrame({'idx_xy':xy_rdaps, 'wspd':value_num})
                    # print(df.head())
                    # print(df.tail())
                elif unit_str == 'percent':
                    df = pd.DataFrame({'idx_xy':xy_rdaps, 'humi':value_num})
                    # print(df.head())
                    # print(df.tail())
                elif unit_str == 'deg':
                    df = pd.DataFrame({'idx_xy':xy_rdaps, 'wdir':value_num})
                    # print(df.head())
                    # print(df.tail())
                elif unit_str == 'C':
                    df = pd.DataFrame({'idx_xy':xy_rdaps, 'temp':value_num})
                    # print(df.head())
                    # print(df.tail())
                df['base'] = base_time
                df['fcst'] = fcst_time

                if type == 'Temp':
                    print(str(l)+"_"+str(time_tgt)+'_'+str(type))
                    data_df = df
                    print(data_df.shape)
                else:
                    print(str(l)+"_"+str(time_tgt)+'_'+str(type))
                    data_df = data_df.merge(df, on=['base','fcst','idx_xy'], how='left')
                    print(data_df.shape)

                rdaps_meta_aggr.append(rdaps_meta)
            except:
                pass
    rdaps_aggr.append(data_df)

rdaps_aggr
rdaps_aggr = pd.concat(rdaps_aggr)
rdaps_meta_aggr = pd.concat(rdaps_meta_aggr)
rdaps_aggr.tail()
rdaps_aggr.tail()
rdaps_aggr.shape
rdaps_aggr.columns

rdaps_aggr['idx_x'] = rdaps_aggr['idx_xy'].apply(lambda x: str(x[0]))
rdaps_aggr['idx_y'] = rdaps_aggr['idx_xy'].apply(lambda x: str(x[1]))
rdaps_aggr['base'] = rdaps_aggr['base'].astype(str)
rdaps_aggr['fcst'] = rdaps_aggr['fcst'].astype(str)
rdaps_aggr['wdir'] = rdaps_aggr['wdir'].astype(str)
rdaps_aggr['rain'] = rdaps_aggr['rain'].astype(str)
rdaps_aggr['humi'] = rdaps_aggr['humi'].astype(str)
rdaps_aggr['wspd'] = rdaps_aggr['wspd'].astype(str)
rdaps_aggr['temp'] = rdaps_aggr['temp'].astype(str)
rdaps_aggr.drop('idx_xy', axis=1, inplace=True)
rdaps_data = rdaps_aggr.drop_duplicates().copy()


rdaps_aggr = rdaps_data.loc[:,['idx_x','idx_y', 'base','fcst','temp','humi','wspd','wdir','rain']]
rdaps_aggr.head()

res = rdaps_aggr.apply(lambda x: rdaps2xy(np.int64(x[0]), np.int64(x[1])), axis=1) #좌표 추출 
rdaps_aggr['lon'] = res.apply(lambda x: str(x[0])) # 데이터베이스 업로드를 위해 문자열로 변환시킴.
rdaps_aggr['lat'] = res.apply(lambda x: str(x[1])) # 데이터베이스 업로드를 위해 문자열로 변환시킴.
rdaps_aggr


with open("D:/pm_data/rdaps_data/rdaps_"+ start_str + '.pkl', 'wb') as dif:
    pickle.dump(rdaps_aggr, dif)

with open("D:/pm_data/rdaps_data/rdaps_meta_"+ start_str + '.pkl', 'wb') as dif:
    pickle.dump(rdaps_meta_aggr, dif)



host = "localhost"
port = "5432"
user = "dataotaku"
password = "!@#qwe5392"
db = "particle"

conn = psql.connect(host=host, port=port, user=user, password=password, dbname=db)
cur = conn.cursor()
cur
# cur.execute("select * from tb_sal_his; ")
# cur.fetchall()

cur.execute("""create table if not exists rdaps_data (idx_x varchar(10), idx_y varchar(10), 
base varchar(20), fcst varchar(20), temp varchar(20), humi varchar(20), wspd varchar(20), wdir varchar(20), rain varchar(20),
lon varchar(20), lat varchar(20),
primary key(idx_x, idx_y, base, fcst));""")

# conn.commit()

for k in range(rdaps_aggr.shape[0]):
    try:
        cur.execute("""insert into rdaps_data
                        (idx_x, idx_y, base, fcst, temp, humi, wspd, wdir, rain, lon, lat) VALUES 
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (tuple(rdaps_aggr.iloc[k,:]))
        )
        conn.commit()
    except:
        pass


cur.close()
conn.close()
