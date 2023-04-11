import py_entitymatching as em
import os
import pandas as pd
import numpy as np
import copy
from collections import defaultdict
import pickle
import joblib
import sys
sys.path.append("..") 
from utils import *

#load training dataset
xstr=['../X2.csv','../X3.csv','../X4.csv']
ystr=['../Y2.csv','../Y3.csv','../Y4.csv']
nstr=['X2_model','X3_model','X4']

for xidx in range(2):
    X=pd.read_csv(xstr[xidx])
    if "price" in X.columns:
        X['price']=X['price'].fillna(0)
    X=X.fillna("")
    Y=pd.read_csv(ystr[xidx])

    brand_list=['lenovo', 'hp', 'acer', 'dell', 'toshiba', 'asus', 'msi', 'alienware', 'sony', 'apple', 'samsung', 'panasonic', 'gateway', 'fuji']
    core_list=['i5 3320m', 'i7 3667u', 'i7 2', 'i5 3427u', 'i5 2520m', 'i7 3520m', 'i7 620m', '2 duo', 'i3 2367m', 'i5 2540m', 'i5 3360m', 'i3 4010u', 'i3 2348m', 'i5 4210u', 'a8 5545m', 'i5 2 ', 'i5 3210m', 'i3 3110m', 'i5 4200u', 'i7 4702mq', 'i7 3667u', 'i5 m520', 'i5 3380m', 'i5 2450m', 'i7 620m,', 'i5 4310u', 'i5 2467m', 'i7 720qm', 'i5 2410m', 'i5 560m', 'i5 3230m', 'i7 2620m', 'i3 3227u','i5 3437u','i7 4500u','i7 m640','i7 3630qm','i5 3317u','p8700','p8600','l9400']
    cpu_list=['intel','amd']
    hp_model_list=['8460p','8540w','8730w','8530w','8440p','nc6400','6930p','8740w','8560p','f009wm','9470m','revolve','8560p','g070nr','2170p','8530p','8570p','2560p', '8770w','8530w','g012dx','8470w','8730w','2570p','d053cl','2760p','dv6000 dv6700','nc6400','d090nr','r150nr','g4 1','2570p d8','2540p','p030nr']
    lenovo_model_list=['x230t','x200t','x201','x130e','x220','x1003','x1 carbon','x120e','x200','x100e','x230']
    thinkpad_special_model_list=['e 300','e 450','e 350','06112','3444cuu','3444','3438','3437','2338','4291','4290','2339','0596','0627','3093','0622','4287','3435','2324','2325','2320', '3460']
    dell_model_list=['3751slv','m731r','5780slv','15001slv','5735','5547','15 7']
    asus_model_list=['ux301la','ux21e']
    acer_model_list=['6607','588','e1 532 29574','2438','e1 572','e5 771','e1 771','v3 572','ux31a','e1 522','e3 111','v3 111p','v3 772','v7 482','e1 731','e1 532','e1 571','e5 57','e5 521','e5 531',
            'm5 481pt','m5 481t','p3 171','5742 6838','s7 392','5620','r7 572','v5 122p','v5 123','v5 132p','v5 573p','as5552']

    core_more_list=[one.replace("i5","").replace("i3","").replace("i7","") for one in core_list]
    core_list=core_list+core_more_list

    X['title']=X['title'].apply(lambda x:x.lower())
    after_brand_set=[]
    X['gram_3']=""
    for i,row in X['title'].iteritems():
        arow=row.replace("-"," ").split()
        new_brand_list=['thinkpad']+brand_list
        for bb in new_brand_list:
            if bb in arow:
                bidx=arow.index(bb)
                content=' '.join(arow[bidx+1:bidx+4])
                X.loc[i,'gram_3']=content
                break
    # cpu_list=['Intel','AMD']
    X['brand_refine']=""
    for i,item in X.iterrows():
        for j,bb in enumerate(brand_list):
            if bb in (item['title']+item['brand']).lower():
                X.loc[i,'brand_refine']=bb
                break
    X['cpu_brand_refine']=""
    for i,item in X.iterrows():
        for j,bb in enumerate(cpu_list):
            if bb in item['cpu_brand']+item['title']:
                X.loc[i,'cpu_brand_refine']=bb
                break
    X['core_refine']=""
    for i,item in X.iterrows():
        core_strs= (' '.join([item['cpu_brand'],item['cpu_type'],item['title']])).lower()
        for j,bb in enumerate(core_list):
            if bb in core_strs.replace("-"," "):
                X.loc[i,'core_refine']=bb
                break
    X['frequency_refine']=""
    for i,item in X.iterrows():
        arow=(item['title']+" "+item['cpu_frequency']).lower().replace('ghz.',"ghz ")
        if 'ghz' in arow:
            row=arow.replace("("," ").split()
            for one in row:
                if 'ghz' == one:
                    idx=row.index('ghz')
                    X.loc[i,'frequency_refine']=''.join(row[idx-1:idx+1])
                    break
                if 'ghz' in one:
                    X.loc[i,'frequency_refine']=one
                    break
    storage_list=['320gb', '500gb', '250gb', '128gb', '750gb', '80gb', '120gb',
        '160gb', '180gb', '300gb', '256gb', '200gb', '60gb', '240gb', '640gb', '1000gb']
    X['storage_refine']=""
    for i,item in X.iterrows():
        row=(item['title']+" " + item['ram_capacity']+" "+item['ssd_capacity']).lower().replace("("," ").replace(" gb","gb").replace("g ","gb ")
        for one in storage_list:
            if one in row:
                X.loc[i,'storage_refine']=one
                break              
    ram_list=[' 4gb', ' 8gb', ' 6gb', ' 3gb', ' 1gb', ' 2gb']
    X['ram_refine']=""
    for i,item in X.iterrows():
        row=(item['title']+" " + item['ram_capacity']).lower().replace("("," ").replace(" gb","gb").replace("g ","gb ")
        for one in ram_list:
            if one in row:
                X.loc[i,'ram_refine']=one[1:]
                break        
    X['model_refine']=""
    model_list=acer_model_list+thinkpad_special_model_list+hp_model_list+lenovo_model_list+asus_model_list+dell_model_list
    for i,item in X.iterrows():
        row=item['title'].replace("-"," ")
        for j,bb in enumerate(model_list):
            if bb in row:
                X.loc[i,'model_refine']=bb
                break            

    
    # X=X[['instance_id','title','cpu_brand_refine','core_refine','frequency_refine','model_refine','ram_refine','storage_refine','brand_refine','gram_3']]
    X=X[['instance_id','title','core_refine','frequency_refine','model_refine','ram_refine','storage_refine','gram_3']]
    X.loc[X['instance_id']=='source15__742','ram_refine']='4gb'
    X.loc[X['instance_id']=='source0__13857','ram_refine']='8gb'
    # X.loc[X['instance_id']=='source0__14091','ram_refine']='8gb'
    X.loc[X['instance_id']=='source0__13857','ram_refine']='8gb'
    X.loc[X['instance_id']=='source0__18025','ram_refine']='8gb'
    X.loc[X['instance_id']=='source0__51745','ram_refine']='6gb'  
    X.loc[X['instance_id']=='source0__14091','frequency_refine']='5ghz'  
    X.loc[X['instance_id']=='source0__14091','storage_refine']='500gb'
    X.loc[X['instance_id']=='source0__14091','ram_refine']='4gb'
    if('source0__16900' in X['instance_id']):
        X.loc[X['instance_id']=='source0__14091','title']=X[X['instance_id']=='source0__16900']['title'].iloc[0]



    label_raw=Y.merge(X,left_on=['left_instance_id'],right_on=['instance_id'])
    label_raw=label_raw.merge(X,left_on=['right_instance_id'],right_on=['instance_id'])
    label_raw['_id']=label_raw.index
    label_raw=label_raw.drop(['instance_id_x','instance_id_y'],axis=1)
    G=label_raw[:]

    #for split traning dataset
    # msk=np.random.rand(len(G))<0.8
    # train_G=G[msk]
    # test_G=G[~msk]
    # G=train_G

    #train the original model
    em.set_key(G,'_id')
    em.set_fk_ltable(G,'left_instance_id')
    em.set_fk_rtable(G,'right_instance_id')
    em.set_key(X,'instance_id')
    feature_X=X.drop(['instance_id'],axis=1)
    feature_table = em.get_features_for_matching(feature_X,feature_X,validate_inferred_attr_types=False)
    em.set_ltable(G,X)
    em.set_rtable(G,X)
    attrs_from_table = [one for one in G.columns if one not in ['left_instance_id','right_instance_id','label','_id']]
    H = em.extract_feature_vecs(G, 
                                feature_table=feature_table, 
                                attrs_before = attrs_from_table,
                                attrs_after='label',
                                show_progress=True)

    # tree rf for extracting blocking rules
    rf=em.RFMatcher(n_estimators=120,class_weight="balanced")
    # rf=em.RFMatcher(n_estimators=10)
    attrs_to_be_excluded = []
    attrs_to_be_excluded.extend(['_id', 'left_instance_id','right_instance_id','label'])
    attrs_to_be_excluded.extend(attrs_from_table)
    rf.fit(table=H, exclude_attrs=attrs_to_be_excluded, target_attr='label')

    #save feature_table
    feature_table.to_csv("model/feature_table_"+nstr[xidx]+".csv")

    #save model
    joblib.dump(rf,"model/random_forest_"+nstr[xidx]+".joblib")

    old_features=feature_table['feature_name']
    with open("temp/old_features_"+nstr[xidx]+".data","wb") as f:
        pickle.dump(old_features,f)

