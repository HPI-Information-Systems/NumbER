import py_entitymatching as em
import os
import pandas as pd
import numpy as np
import copy
from collections import defaultdict
import pickle
import joblib
import itertools
import re
import sys
sys.path.append("..") 
from utils import *
#load training dataset

X=pd.read_csv("../X4.csv")
X['price']=X['price'].fillna(0)
X=X.fillna("")
Y=pd.read_csv("../Y4.csv")
X.columns=['name_', 'price', 'brand', 'size_', 'instance_id']
X['name_']=X['name_'].apply(lambda x:x.lower())
s=str.maketrans("(|)/&!\"#$?%+,;:","               ")
X['name_']=X['name_'].apply(lambda x:x.translate(s))
X['name_']=X['name_'].apply(lambda x:x.replace("mo ",'mb '))
X['name_']=X['name_'].apply(lambda x:x.replace(" mb ",'mb '))
X['name_']=X['name_'].apply(lambda x:x.replace("go ",'gb '))
X['name_']=X['name_'].apply(lambda x:x.replace(" gb ",'gb '))
X['name_']=X['name_'].apply(lambda x:x.replace(" gb",'gb'))
X['name_']=X['name_'].apply(lambda x:x.replace(" usb 3",' usb3'))
X['name_']=X['name_'].apply(lambda x:x.replace(" usb 2",' usb2'))
X['name_']=X['name_'].apply(lambda x:x.replace(" classe ",' class '))
X['name_']=X['name_'].apply(lambda x:x.replace(" clase ",' class '))
X['name_']=X['name_'].apply(lambda x:x.replace("\\",' '))
X['name_']=X['name_'].apply(lambda x:x.replace("attaché",'attache'))

id2Ins=defaultdict(Instance)
Y_match=Y[Y['label']==1]
for one in set(list(Y_match['left_instance_id'])+list(Y_match['right_instance_id'])):
    oIns=Instance(one)
    id2Ins[one]=oIns
all_instance=[]
for _,row in Y_match.iterrows():
    lid=row['left_instance_id']
    rid=row['right_instance_id']
    xx=id2Ins[lid]
    yy=id2Ins[rid]
    union(xx,yy)
    all_instance.append(xx)
    all_instance.append(yy)
clusters=defaultdict(list)
for one in all_instance:
    clusters[find(one)._id].append(one._id)
clusters1=[list(set(clusters[one])) for one in clusters]
for idx,i in enumerate(clusters1):
    for j in i:
        X.loc[X['instance_id']==j,'group']=idx
    

X['card_type']=""
usb_list=['usb 2','type c','typec','type a','typea','usb 3','3.1','3.0','transmemory','pen drive']
micro_list=['microsdhc','microsdxc','microsd']
sdhc_list=['sdhc','sdxc','class10']
for i,row in X['name_'].iteritems():
    sign=0
    for aa in usb_list:
        if aa in row:
            X.loc[i,'card_type']='usb'
            sign=1
            break
    if(sign):
        continue
    row=row.replace('classe','class').replace("class ",'class').replace('cl ','cl')
    arow=''.join(row.split())

    for aa in micro_list:
        if aa in arow:
            X.loc[i,"card_type"]='micro'
            sign=1
            break
    if(sign):
        continue
    if('usb' in row):
        X.loc[i,'card_type']='usb'
X['sdhc_type']=""
for i,row in X.iterrows():
    if row['card_type']=="":
        if 'micro' not in row['name_']:
            for bb in ['sdhc','sdxc','secure']:
                if bb in row['name_']:
                    X.loc[i,'sdhc_type']='sdhc'
        
X['first_symbol']=""
for i,item in X['name_'].iteritems():
    item=item.replace("-","")
    item= item.split()
    digits=[one for one in item if re.match('.*\d+', one)]
    for one in digits:
        if one.isdigit() and len(one)<3:
            continue
        if re.match('.*gb$',one):
            continue
        if re.match('.*g$',one):
            continue
        if re.match('.*mb$',one):
            continue
        if re.match('^u.*',one) and len(one)<3:
            continue
        if re.match('^uhs.*',one):
            continue
        if re.match('^usb.*',one):
            continue
        if re.match('^class.*',one):
            continue
        if(one in ['128','256','512','4k','3.0','3.1','2.0']):
            continue
        X.loc[i,'first_symbol']=one
        break
useful_word_list=['microsdhc','microsdxc',' sdhc','u3','u1 ','class10','class3','class4','uhsi ','uhsii ',' sdxc',
                  'usb2.0','usb3.0','usb3.1','typec','microsd ','ssd ','typea']
X['useful_word']=""
for i,item in X['name_'].iteritems():
    item=item.replace("-","")
    item=item.replace("class ","class")
    item=item.replace(" cl "," class")
    item=item.replace("micro sdhc","microsdhc")
    item=item.replace("cl10",'class10')
    item=item.replace("uhs ","uhs")
    item=item.replace("type c","typec")
    item=item.replace("type a","typea")
    temp=[one for one in useful_word_list if one in item]
    X.loc[i,'useful_word']=' '.join(temp)


toshiba_model_list=['u302','u201','u202','n401','n302','m302','m401','n101']
toshiba_useful_word=['exceria','pro']
toshiba_X=X[X['brand']=='TOSHIBA'].copy()
toshiba_X['brand']='TOSHIBA'
toshiba_X['model_refine']=""
for i,item in toshiba_X['name_'].iteritems():
    if('exceria pro') in item:
        item+=' n101 '
    item=item.replace('xpro32uhs2',' n101 ')
    item=item.replace('exceria pro','exceriapro')
    for bb in toshiba_model_list:
        if bb in item:
            toshiba_X.loc[i,'model_refine']=bb
            break
    for bb in toshiba_useful_word:
        if bb in item:
            toshiba_X.loc[i,'useful_word']+=(' '+bb)
toshiba_X.loc[toshiba_X['instance_id']=='altosight.com//3595','model_refine']='n401'    
sony_model_list=['srg1uxa','sf8u','usm8gqx','usm16ca1','sf16n4','usm32gr','usm32gqx','usm128gqx','sl1']
sony_useful_word=['compact','dur','disque']
sony_X=X[X['brand']=='SONY'].copy()
sony_X['model_refine']=""
sony_X['brand']="SONY"
for i,item in sony_X['name_'].iteritems():
    item=item.replace('sl 1',' sl1 ')
    for bb in sony_model_list:
        if bb in item:
            sony_X.loc[i,'model_refine']=bb
            break
    for bb in sony_useful_word:
        if bb in item:
            sony_X.loc[i,'useful_word']+=(' '+bb)
kingston_model_list=['101','sda10','sdca3','sda3','se9','g4','g3','hxs3']
kingston_useful_word=['datatraveler','ultimate','g2','savage','hyper']
kingston_X=X[X['brand']=='Kingston'].copy()
kingston_X['model_refine']=""
kingston_X['brand']="Kingston"
for i,item in kingston_X['name_'].iteritems():
    item=item.replace('uhs class 1','uhsi')
    item=item.replace('data traveler','datatraveler')
    for bb in kingston_model_list:
        if bb in item:
            kingston_X.loc[i,'model_refine']=bb
            break
    for bb in kingston_useful_word:
        if bb in item:
            kingston_X.loc[i,'useful_word']+=(' '+bb)
    for bb in ['hyperx','savage','hxs']:
        if bb in item:
            kingston_X.loc[i,'model_refine']='hxs3'
intenso_model_list=['rainbow','premium','basic','speed']
intenso_useful_word=['speed','basic','rainbow','premium','3502450','3503470','3534460','3530460','3534490']
intenso_X=X[X['brand']=='Intenso'].copy()
intenso_X['model_refine']=""
intenso_X['brand']="Intenso"
for i,item in intenso_X['name_'].iteritems():
    item=item.replace('uhs class 1','uhsi')
    item=item.replace('data traveler','datatraveler')
    for bb in intenso_model_list:
        if bb in item:
            intenso_X.loc[i,'model_refine']=bb
            break
    for bb in intenso_useful_word:
        if bb in item:
            intenso_X.loc[i,'useful_word']+=(' '+bb)
    if('3503470' in item and intenso_X.loc[i,'model_refine']==""):
        intenso_X.loc[i,'model_refine']="basic"
    if('3502450' in item and intenso_X.loc[i,'model_refine']==""):
        intenso_X.loc[i,'model_refine']="rainbow"
    if('3534460' in item and intenso_X.loc[i,'model_refine']==""):
        intenso_X.loc[i,'model_refine']="premium"
    if('3530460' in item and intenso_X.loc[i,'model_refine']==""):
        intenso_X.loc[i,'model_refine']="speed"
    if('3534490' in item and intenso_X.loc[i,'model_refine']==""):
        intenso_X.loc[i,'model_refine']="premium"
intenso_X.loc[intenso_X['instance_id']=="altosight.com//8124",'model_refine']='premium'
intenso_X.loc[intenso_X['instance_id']=="altosight.com//8127",'model_refine']='premium'
pny_model_list=['fd8','fd32','fd16','fd128']
pny_useful_word=['datatraveler','ultimate','g2','savage','attache4']
pny_X=X[X['brand']=='PNY'].copy()
pny_X['model_refine']=""
pny_X['brand']="PNY"
for i,item in pny_X['name_'].iteritems():
    item=item.replace('uhs class 1','uhsi')
    item=item.replace('data traveler','datatraveler')
    item=item.replace('attache 4','attache4')
    for bb in pny_model_list:
        if bb in item:
            pny_X.loc[i,'model_refine']=bb
            break
    for bb in pny_useful_word:
        if bb in item:
            pny_X.loc[i,'useful_word']+=(' '+bb)
    for bb in ['hyperx','savage','hxs']:
        if bb in item:
            pny_X.loc[i,'model_refine']='hxs3'
transcend_model_list=['ultimate','extreme']
transcend_useful_word=['datatraveler','ultimate','g2','savage','attache4']
transcend_X=X[X['brand']=='Transcend'].copy()
transcend_X['model_refine']=""
transcend_X['brand']="Transcend"
for i,item in transcend_X['name_'].iteritems():
    item=item.replace('uhs class 1','uhsi')
    item=item.replace('data traveler','datatraveler')
    item=item.replace('attache 4','attache4')
    for bb in transcend_model_list:
        if bb in item:
            transcend_X.loc[i,'model_refine']=bb
            break
    for bb in transcend_useful_word:
        if bb in item:
            transcend_X.loc[i,'useful_word']+=(' '+bb)
    for bb in ['hyperx','savage','hxs']:
        if bb in item:
            transcend_X.loc[i,'model_refine']='hxs3'
sandisk_model_list=['ultraplus','extremeplus','glide','fit','extreme','ultra']
sandisk_useful_word=['cruzer','ultimate','dualdrive','dual','pro','extreme','plus']
sandisk_X=X[X['brand']=='SANDISK'].copy()
sandisk_X['model_refine']=""
sandisk_X['brand']="SANDISK"
for i,item in sandisk_X['name_'].iteritems():
    item=item.replace('uhs class 1','uhsi')
    item=item.replace('data traveler','datatraveler')
    item=item.replace('attache 4','attache4')
    item=item.replace('dual drive','dualdrive')
    item=item.replace("extrem ",'extreme ')
    item=item.replace("ultra plus",'ultraplus')
    item=item.replace("extreme plus",'extremeplus')
    item=item.replace("ext ",'extremeplus')
    temp_model_list=sandisk_model_list
    if X.loc[i,'size_']=='8 GB':
        temp_model_list.append('cruzer')
    for bb in temp_model_list:
        if bb in item:
            sandisk_X.loc[i,'model_refine']=bb
            break
    for bb in sandisk_useful_word:
        if bb in item:
            sandisk_X.loc[i,'useful_word']+=(' '+bb)
    for bb in ['hyperx','savage','hxs']:
        if bb in item:
            sandisk_X.loc[i,'model_refine']='hxs3'
sandisk_X.loc[sandisk_X['instance_id']=="altosight.com//776",'model_refine']='ultra'
sandisk_X.loc[sandisk_X['instance_id']=="altosight.com//777",'model_refine']='ultra'
sandisk_X.loc[sandisk_X['instance_id']=="altosight.com//909",'model_refine']='ultra'
sandisk_X.loc[sandisk_X['instance_id']=="altosight.com//12344",'model_refine']='ultra'
lexar_model_list=['1400','300','v10','v30','s25','s30','s70','p20','c20c','c20m']
lexar_useful_word=['premium','xqd','professional','16gabeu']
lexar_X=X[X['brand']=='LEXAR'].copy()
lexar_X['model_refine']=""
lexar_X['brand']="LEXAR"
for i,item in lexar_X['name_'].iteritems():
    item=item.replace('uhs class 1','uhsi')
    item=item.replace('data traveler','datatraveler')
    item=item.replace('platinum','premium')
    for bb in lexar_model_list:
        if bb in item:
            lexar_X.loc[i,'model_refine']=bb
            break
    for bb in lexar_useful_word:
        if bb in item:
            lexar_X.loc[i,'useful_word']+=(' '+bb)
    if 'jumpdrive' in item and lexar_X.loc[i,'card_type']=="":
        lexar_X.loc[i,'card_type']='usb'   
lexar_X.loc[lexar_X['instance_id']=="altosight.com//1198",'model_refine']='s70'
lexar_X.loc[lexar_X['instance_id']=="altosight.com//13459",'model_refine']='c20c'
lexar_X.loc[lexar_X['instance_id']=="altosight.com//4073",'model_refine']='c20c'
samsung_model_list=['a10','a20','a30','a50','512gb','t3','t5','t7','a7','j4','j6']
samsung_useful_word=['black','blue','red','white','ssd','dur','pro','evo','plus']
samsung_color_list=['black','blue','red','white']
samsung_X=X[X['brand']=='Samsung'].copy()
samsung_X['model_refine']=""
samsung_X['brand']="samsung"
for i,item in samsung_X['name_'].iteritems():
    item=item.replace('uhs class 1','uhsi')
    item=item.replace('data traveler','datatraveler')
    item=item.replace('platinum','premium')
    for bb in samsung_model_list:
        if bb in item:
            samsung_X.loc[i,'model_refine']=bb
            break
    for bb in samsung_color_list:
        if bb in item:
            samsung_X.loc[i,'model_refine']+=(" "+bb)
            break
    for bb in samsung_useful_word:
        if bb in item:
            samsung_X.loc[i,'useful_word']+=(' '+bb)
    if 'evo' in item and samsung_X.loc[i,'size_']=="16 GB":
        samsung_X.loc[i,'model_refine']='evo'
samsung_X.loc[samsung_X['instance_id']=="altosight.com//1129",'card_type']='micro'

def applyMergeX4(X_data):
    X_data=X_data[['name_','brand', 'size_', 'instance_id', 'card_type', 'sdhc_type',
    'first_symbol', 'useful_word', 'model_refine']].reset_index(drop=True)
    candidate_list=X_data.groupby(['brand','size_']).apply(lambda x:x['instance_id'].to_list()).to_list()
    lcol=[]
    rcol=[]
    for one in candidate_list:
        createResult(one,lcol,rcol)
    merge_data=pd.DataFrame({'left_instance_id':lcol,'right_instance_id':rcol})
    merge_data=merge_data.merge(X_data,left_on='left_instance_id',right_on='instance_id')
    merge_data=merge_data.merge(X_data,left_on='right_instance_id',right_on='instance_id')
    merge_data.drop(['instance_id_x','instance_id_y'],axis=1,inplace=True)
    merge_data['brand']=merge_data['brand_x']
    merge_data.drop(['brand_x','brand_y'],axis=1,inplace=True)
    merge_data['size_']=merge_data['size__x']
    merge_data.drop(['size__x','size__y'],axis=1,inplace=True)
    return merge_data

brand_df=[applyMergeX4(one) for one in [lexar_X,sandisk_X,intenso_X,toshiba_X,sony_X,transcend_X,kingston_X,samsung_X]]

Xid=X.set_index("instance_id")
for odf in brand_df:
    odf['label']=0
    for i,row in odf[:].iterrows():
        lid=row['left_instance_id']
        rid=row['right_instance_id']
        if(Xid.loc[lid,'group']==Xid.loc[rid,'group']):
            odf.loc[i,'label']=1

def func_match(x,feat):
    feat_x=feat+"_x"
    feat_y=feat+"_y"
    if x[feat_x]==x[feat_y]=="":
        return 0
    if x[feat_x]==x[feat_y]:
        return 1
    elif x[feat_x]=="" or x[feat_y]=="":
        return -0.5
    else:
        return -1
def word_match(x,feat):
    feat_x=feat+"_x"
    feat_y=feat+"_y"
    lword=x[feat_x].split()
    rword=x[feat_y].split()
    overlap_words=[one for one in lword if one in rword]
    score=0
    for bb in overlap_words:
        if bb in useful_word_list:
            score+=1
        else:
            score+=2
    if(len(lword)>0 and len(rword)>0 and len(overlap_words)==0):
        score=-2
    return score
refine_X=pd.concat([lexar_X,sandisk_X,intenso_X,toshiba_X,sony_X,transcend_X,kingston_X,samsung_X])

brand_list=["LEXAR",'SANDISK','Intenso','TOSHIBA','SONY','Transcend','Kingston','samsung']

for iii,odf in enumerate(brand_df):
    G=odf.copy()
    G['_id']=range(len(G))
    em.set_key(G,'_id')
    em.set_fk_ltable(G,'left_instance_id')
    em.set_fk_rtable(G,'right_instance_id')
    em.set_key(refine_X,'instance_id')
    feature_X=refine_X[['name_','model_refine','card_type','sdhc_type','first_symbol','useful_word']]
    feature_table = em.get_features_for_matching(feature_X,feature_X,validate_inferred_attr_types=False)
    em.set_ltable(G,refine_X)
    em.set_rtable(G,refine_X)
    attrs_from_table = [one for one in G.columns if one not in ['left_instance_id','right_instance_id','label','_id']]
    for i in range(len(G)):
        ratevalue=0.3
        if(brand_list[iii]=='LEXAR'):
            ratevalue=0.01
        if(np.random.rand()<ratevalue):
            G.loc[i,'model_refine_x']=""
            G.loc[i,'model_refine_y']=""

        if(brand_list[iii]=='SANDISK'):
            if(np.random.rand()<0.01):
                G.loc[i,'model_refine_x']='ultra'
            if(np.random.rand()<0.01):
                G.loc[i,'model_refine_y']='ultra'
            if(np.random.rand()<0.01):
                G.loc[i,'model_refine_x']='extreme'
            if(np.random.rand()<0.01):
                G.loc[i,'model_refine_y']='extreme'

    H = em.extract_feature_vecs(G, 
                                feature_table=feature_table, 
                                attrs_before = attrs_from_table,
                                attrs_after='label',
                                show_progress=True)

#     em.set_ltable(GG,refine_X)
#     em.set_rtable(GG,refine_X)
#     em.set_fk_ltable(GG,'left_instance_id')
#     em.set_fk_rtable(GG,'right_instance_id')
#     em.set_key(GG,'_id')
#     HH = em.extract_feature_vecs(GG, 
#                                 feature_table=feature_table, 
#                                 attrs_before = attrs_from_table,
#                                 attrs_after='label',
#                                 show_progress=True)
    H['match_card_type']=H.apply(lambda x:func_match(x,'card_type'),axis=1)
    H['match_model_refine']=H.apply(lambda x:func_match(x,'model_refine'),axis=1)
    H['match_sdhc_type']=H.apply(lambda x:func_match(x,'sdhc_type'),axis=1)
    H['match_first_symbol']=H.apply(lambda x:func_match(x,'first_symbol'),axis=1)
    H['match_useful_word']=H.apply(lambda x:word_match(x,'useful_word'),axis=1)

    rf=em.RFMatcher(n_estimators=50,class_weight='balanced')
    attrs_to_be_excluded = []
    attrs_to_be_excluded.extend(['_id', 'left_instance_id','right_instance_id','label'])
    attrs_to_be_excluded.extend(attrs_from_table)
    rf.fit(table=H, exclude_attrs=attrs_to_be_excluded, target_attr='label')
    joblib.dump(rf,"model/X4_model/random_forest_X4_"+brand_list[iii]+".joblib")

#save feature_table
feature_table.to_csv("model/feature_table_X4_model.csv")


