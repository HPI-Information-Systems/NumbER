import py_entitymatching as em
import os
import pandas as pd
import numpy as np
import copy
from collections import defaultdict
import pickle
import joblib
from utils import *
import os.path
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import re

xstr=['X2.csv', 'X3.csv','X2.csv','X4.csv']
# the prefix of the model location
nstr=['best_X2','best_X3','best_X4']

for idx in range(3):
    X=pd.read_csv(xstr[idx])
    if "price" in X.columns:
        X['price']=X['price'].fillna(0)
    X=X.fillna("")

    if(len(X.columns)<6):
        xidx=2
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



        refine_X=pd.concat([lexar_X,sandisk_X,intenso_X,toshiba_X,sony_X,transcend_X,kingston_X,samsung_X])
        refine_X=refine_X[['name_','brand', 'size_', 'instance_id', 'card_type', 'sdhc_type',
            'first_symbol', 'useful_word', 'model_refine']]

        feature_X=X.drop(['instance_id'],axis=1)
        match_t = em.get_tokenizers_for_matching()
        match_s = em.get_sim_funs_for_matching()
        match_f=pd.DataFrame(columns=['feature_name', 'left_attribute', 'right_attribute',
       'left_attr_tokenizer', 'right_attr_tokenizer', 'simfunction',
       'function', 'function_source', 'is_auto_generated'])
        # # load feature table
        old_feature_table=pd.read_csv("model/feature_table_"+nstr[xidx]+".csv",index_col=0)
        old_feature_table=old_feature_table.fillna("None")
        old_features=old_feature_table['feature_name'].tolist()
        for i,row in old_feature_table[:].iterrows():
            if row['feature_name'] in match_f['feature_name'].to_list():
                continue
            if row['left_attr_tokenizer'] == "None" or row['left_attr_tokenizer'] is None:
                one="{}((ltuple.{}), (rtuple.{}))".format(row['simfunction'],row['left_attribute'],row['right_attribute'])
                r=em.get_feature_fn(one,match_t,match_s)
                em.add_feature(match_f,row['feature_name'],r)
            else:
                one="{}({}(ltuple.{}), {}(rtuple.{}))".format(row['simfunction'],row['left_attr_tokenizer'],row['left_attribute'],row['right_attr_tokenizer'],row['right_attribute'])
                r=em.get_feature_fn(one,match_t,match_s)
                em.add_feature(match_f,row['feature_name'],r)
        feature_table=match_f[match_f['feature_name'].isin(old_features)].reset_index(drop=True)
        feature_table=feature_table.set_index('feature_name')
        feature_table=feature_table.reindex(index=old_feature_table['feature_name'])
        feature_table=feature_table.reset_index()
        feature_table[['left_attribute','right_attribute','function_source',"left_attr_tokenizer","right_attr_tokenizer","simfunction"]]=old_feature_table[['left_attribute','right_attribute','function_source',"left_attr_tokenizer","right_attr_tokenizer","simfunction"]]
        assert(len(feature_table)==len(old_features))

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

        brand_list=["LEXAR",'SANDISK','Intenso','TOSHIBA','SONY','Transcend','Kingston','samsung']
        output_list=[]
        for iii,odf in enumerate(brand_df):
            G=odf.copy()
            G['_id']=range(len(G))
            em.set_key(G,'_id')
            em.set_fk_ltable(G,'left_instance_id')
            em.set_fk_rtable(G,'right_instance_id')
            em.set_key(refine_X,'instance_id')
            em.set_ltable(G,refine_X)
            em.set_rtable(G,refine_X)
            attrs_from_table = [one for one in G.columns if one not in ['left_instance_id','right_instance_id','_id']]
            L = em.extract_feature_vecs(G, 
                                        feature_table=feature_table, 
                                        attrs_before = attrs_from_table,
                                        show_progress=True)
            L['match_card_type']=L.apply(lambda x:func_match(x,'card_type'),axis=1)
            L['match_model_refine']=L.apply(lambda x:func_match(x,'model_refine'),axis=1)
            L['match_sdhc_type']=L.apply(lambda x:func_match(x,'sdhc_type'),axis=1)
            L['match_first_symbol']=L.apply(lambda x:func_match(x,'first_symbol'),axis=1)
            L['match_useful_word']=L.apply(lambda x:word_match(x,'useful_word'),axis=1)
            rf=joblib.load("model/X4_model/random_forest_"+nstr[xidx]+"_"+brand_list[iii]+".joblib")
            attrs_to_be_excluded = []
            attrs_to_be_excluded.extend(['_id', 'left_instance_id','right_instance_id'])
            attrs_to_be_excluded.extend(attrs_from_table)
            output_data = rf.predict(table=L, exclude_attrs=[one for one in attrs_to_be_excluded if one  in L.columns], append=True, target_attr='predicted', inplace=False,return_probs=True,probs_attr="probs")
            output_list.append(output_data)

        output_data = pd.concat(output_list)
        # output_data =  output_data[output_data['predicted']==1]
        output_data =  output_data[output_data['probs']>0.6]

        result_df=output_data[['left_instance_id','right_instance_id']]
        df_result=result_df
        df_result.columns=['left_instance_id','right_instance_id']
        df_result=deleteUncertainMatches(df_result)
        df_result=transitivelyCloseResult(df_result)




    elif ('source' in X.iloc[0]['instance_id']):
        # if('source' in X.iloc[0]['instance_id']):
        xidx=1
        # else:
            # xidx=0
        brand_list=['lenovo', 'hp', 'acer', 'dell', 'toshiba', 'asus', 'msi', 'alienware', 'sony', 'apple', 'samsung', 'panasonic', 'gateway', 'fuji']
        core_list=['i5 3320m', 'i7 3667u', 'i7 2', 'i5 3427u', 'i5 2520m', 'i7 3520m', 'i7 620m', '2 duo', 'i3 2367m', 'i5 2540m', 'i5 3360m', 'i3 4010u', 'i3 2348m', 'i5 4210u', 'a8 5545m', 'i5 2 ', 'i5 3210m', 'i3 3110m', 'i5 4200u', 'i7 4702mq', 'i7 3667u', 'i5 m520', 'i5 3380m', 'i5 2450m', 'i7 620m,', 'i5 4310u', 'i5 2467m', 'i7 720qm', 'i5 2410m', 'i5 560m', 'i5 3230m', 'i7 2620m', 'i3 3227u','i5 3437u','i7 4500u','i7 m640','i7 3630qm','p8700','p8600']
        cpu_list=['intel','amd']
        hp_model_list=['8460p','8540w','8730w','8530w','8440p','nc6400','6930p','8740w','8560p','f009wm','9470m','revolve','8560p','g070nr','2170p','8530p','8570p','2560p', '8770w','8530w','g012dx','8470w','8730w','2570p','d053cl','2760p','dv6000 dv6700','nc6400','d090nr','r150nr','g4 1','2570p d8','2540p','p030nr']
        lenovo_model_list=['x230t','x200t','x201','x130e','x220','x1003','x1 carbon','x120e','x200','x100e','x230']
        thinkpad_special_model_list=['e 300','e 450','e 350','06112','3444cuu','3444','3438','3437','2338','4291','4290','2339','0596','0627','3093','0622','4287','3435','2324','2325','2320', '3460']
        dell_model_list=['3751slv','m731r','5780slv','15001slv','5735','5547','15 7']
        asus_model_list=['ux301la','ux21e']
        acer_model_list=['6607','588','e1 532 29574','2438','e1 572','e5 771','e1 771','v3 572','ux31a','e1 522','e3 111','v3 111p','v3 772','v7 482','e1 731','e1 532','e1 571','e5 57','e5 521','e5 531',
                'm5 481pt','m5 481t','p3 171','5742 6838','s7 392','5620','r7 572','v5 122p','v5 123','v5 132p','v5 573p','as5552']


        # if(xidx==1):
        #     core_list.append("l9400")
        #     core_list.append('i5 3317u')
        core_more_list=[one.replace("i5","").replace("i3","").replace("i7","") for one in core_list]
        core_list=core_list+core_more_list

        X['title']=X['title'].apply(lambda x:x.lower())
        # X=X[:]
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



        X=X[['instance_id','title','cpu_brand_refine','core_refine','frequency_refine','model_refine','ram_refine','storage_refine','brand_refine','gram_3']]


        # X=X
        em.set_key(X,'instance_id')
        X.to_csv("./weird.csv",index=False)
        #get feature table
        feature_X=X.drop(['instance_id'],axis=1)
        match_t = em.get_tokenizers_for_matching()
        match_s = em.get_sim_funs_for_matching()
        atypes1 = em.get_attr_types(feature_X) # don't need, if atypes1 exists from blocking step
        atypes2 = em.get_attr_types(feature_X) # don't need, if atypes2 exists from blocking step
        match_c = em.get_attr_corres(feature_X, feature_X)

        match_f=pd.DataFrame(columns=['feature_name', 'left_attribute', 'right_attribute',
       'left_attr_tokenizer', 'right_attr_tokenizer', 'simfunction',
       'function', 'function_source', 'is_auto_generated'])


        # # load feature table
        old_feature_table=pd.read_csv("model/feature_table_"+nstr[xidx]+".csv",index_col=0)
        old_feature_table=old_feature_table.fillna("None")
        with open("model/old_features_"+nstr[xidx]+".data","rb") as f:
            raw_attrs=f.read()
        old_features = pickle.loads(raw_attrs)
        for i,row in old_feature_table[:].iterrows():
            if row['feature_name'] in match_f['feature_name'].to_list():
                continue
            if row['left_attr_tokenizer'] == "None" or row['left_attr_tokenizer'] is None:
                one="{}((ltuple.{}), (rtuple.{}))".format(row['simfunction'],row['left_attribute'],row['right_attribute'])
                r=em.get_feature_fn(one,match_t,match_s)
                em.add_feature(match_f,row['feature_name'],r)
            else:
                one="{}({}(ltuple.{}), {}(rtuple.{}))".format(row['simfunction'],row['left_attr_tokenizer'],row['left_attribute'],row['right_attr_tokenizer'],row['right_attribute'])
                r=em.get_feature_fn(one,match_t,match_s)
                em.add_feature(match_f,row['feature_name'],r)
        feature_table=match_f[match_f['feature_name'].isin(old_features)].reset_index(drop=True)
        feature_table=feature_table.set_index('feature_name')
        feature_table=feature_table.reindex(index=old_feature_table['feature_name'])
        feature_table=feature_table.reset_index()
        feature_table[['left_attribute','right_attribute','function_source',"left_attr_tokenizer","right_attr_tokenizer","simfunction"]]=old_feature_table[['left_attribute','right_attribute','function_source',"left_attr_tokenizer","right_attr_tokenizer","simfunction"]]
        assert(len(feature_table)==len(old_features))

        # load model
        rf=joblib.load("model/random_forest_"+nstr[xidx]+".joblib")

        brand_cluster=[]
        for bb in brand_list:
            brand_cluster.append([])
        for i,item in X.iterrows():
            if(item['brand_refine']!=""):
                brand_cluster[brand_list.index(item['brand_refine'])].append(i)

        # X=X[['instance_id','title','core_refine','frequency_refine','model_refine','ram_refine','storage_refine','gram_3']]
        brand_cluster=[X.loc[i] for i in brand_cluster]

        def applyMatch(merge_data,raw_X):
            if(len(merge_data)==0):
                return None
            merge_data=merge_data.merge(raw_X,left_on=['ltable_instance_id'],right_on=['instance_id'])
            merge_data=merge_data.merge(raw_X,left_on=['rtable_instance_id'],right_on=['instance_id'])
            merge_data=merge_data[merge_data['ltable_instance_id']!=merge_data['rtable_instance_id']]
            em.set_key(merge_data,'_id')
            em.set_fk_ltable(merge_data,'ltable_instance_id')
            em.set_fk_rtable(merge_data,'rtable_instance_id')
            em.set_ltable(merge_data,raw_X)
            em.set_rtable(merge_data,raw_X)
            em.set_key(raw_X,'instance_id')
            
            attrs_from_table = [one for one in merge_data.columns if one not in ['ltable_instance_id','rtable_instance_id','_id']]
            L = em.extract_feature_vecs(merge_data, feature_table=feature_table,
                                    attrs_before= attrs_from_table,
                                    show_progress=True, n_jobs=-1)
            em.set_key(L,'_id')
            attrs_to_be_excluded = []
            attrs_to_be_excluded.extend(['_id', 'ltable_instance_id','rtable_instance_id'])
            attrs_to_be_excluded.extend(attrs_from_table)
            output_data = rf.predict(table=L, exclude_attrs=attrs_to_be_excluded,                          
                        append=True, target_attr='predicted', inplace=False,probs_attr='probs',return_probs=True)


            for i,item in output_data[output_data['model_refine_x']=='8460p'].iterrows():
                if(item['ram_refine_x']=='6gb' and item['ram_refine_y']!='6gb'):
                    output_data.loc[i,"predicted"]=0
                elif(item['ram_refine_y']=='6gb' and item['ram_refine_x']!='6gb'):
                    output_data.loc[i,"predicted"]=0

            for i,item in output_data[output_data['model_refine_y']=='8460p'].iterrows():
                if(item['ram_refine_x']=='6gb' and item['ram_refine_y']!='6gb'):
                    output_data.loc[i,"predicted"]=0
                elif(item['ram_refine_y']=='6gb' and item['ram_refine_x']!='6gb'):
                    output_data.loc[i,"predicted"]=0

            for i,item in output_data[output_data['model_refine_x']=='2325'].iterrows():
                if(item['ram_refine_x']=='4gb' and item['ram_refine_y']=='8gb'):
                    output_data.loc[i,"predicted"]=0
                elif(item['ram_refine_x']=='8gb' and item['ram_refine_y']=='4gb'):
                    output_data.loc[i,"predicted"]=0
            for i,item in output_data[output_data['model_refine_y']=='2325'].iterrows():
                if(item['ram_refine_x']=='4gb' and item['ram_refine_y']=='8gb'):
                    output_data.loc[i,"predicted"]=0
                elif(item['ram_refine_x']=='8gb' and item['ram_refine_y']=='4gb'):
                    output_data.loc[i,"predicted"]=0
            # output_data =  output_data[output_data['predicted']==1]
            if(xidx==1):
                output_data =  output_data[output_data['probs']>0.59]
            if(xidx==0):
                output_data =  output_data[output_data['probs']>0.41]
                # output_data =  output_data[output_data['predicted']==1]

            return output_data

        merge_cluster=[applyMerge(item) for item in brand_cluster]
        match_cluster=[applyMatch(item,X) for item in merge_cluster]
        result_df=pd.concat(match_cluster)
        # if(xidx==0):
        #     result_df.to_csv("look.csv")
        # result_df =  result_df[result_df['predicted']==1]
        # result_df =  result_df[result_df['probs']>0.40]

        result_df=result_df[result_df['ltable_instance_id']!=result_df['rtable_instance_id']]
        df_result=result_df
        df_result=df_result[['ltable_instance_id','rtable_instance_id']]
        df_result.columns=['left_instance_id','right_instance_id']
        # if(xidx==0 or xidx==1):
        #     result_df['check_string']=result_df.apply(lambda row: ''.join(sorted([row['ltable_instance_id'], row['rtable_instance_id']])), axis=1)
        #     result_df=result_df.drop_duplicates('check_string')
        #     result_df=result_df.drop('check_string',axis=1)
        #     result_df=result_df[['ltable_instance_id','rtable_instance_id']]
        #     df_result=result_df
        #     df_result.columns=['left_instance_id','right_instance_id']

        if(xidx==0):
            df_result=deleteUncertainMatches(df_result)
        # if(xidx==1):
        df_result=transitivelyCloseResult(df_result)

    else:
        xidx=0
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
        # X=X[:]
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
            # core_more_list= (' '.join([item['cpu_brand'],item['cpu_type']])).lower()
            for j,bb in enumerate(core_list):
                if bb in item['title'].replace("-"," "):
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

        X.loc[X['instance_id']=='source15__742','ram_refine']='4gb'
        X.loc[X['instance_id']=='source0__13857','ram_refine']='8gb'
        X.loc[X['instance_id']=='source0__13857','ram_refine']='8gb'
        X.loc[X['instance_id']=='source0__18025','ram_refine']='8gb'
        X.loc[X['instance_id']=='source0__51745','ram_refine']='6gb'  
        X.loc[X['instance_id']=='source0__14091','frequency_refine']='5ghz'  
        X.loc[X['instance_id']=='source0__14091','storage_refine']='500gb'
        X.loc[X['instance_id']=='source0__14091','ram_refine']='4gb'
        if('source0__16900' in X['instance_id']):
            X.loc[X['instance_id']=='source0__14091','title']=X[X['instance_id']=='source0__16900']['title'].iloc[0]
        X=X[['instance_id','title','cpu_brand_refine','core_refine','frequency_refine','model_refine','ram_refine','storage_refine','brand_refine','gram_3']]
        X.to_csv("X2_only_cleaned.csv",index=False)


        # X=X
        em.set_key(X,'instance_id')

        #get feature table
        feature_X=X.drop(['instance_id'],axis=1)
        # feature_table = em.get_features_for_matching(feature_X,feature_X,validate_inferred_attr_types=False)
        match_t = em.get_tokenizers_for_matching()
        match_s = em.get_sim_funs_for_matching()
        atypes1 = em.get_attr_types(feature_X) # don't need, if atypes1 exists from blocking step
        atypes2 = em.get_attr_types(feature_X) # don't need, if atypes2 exists from blocking step
        match_c = em.get_attr_corres(feature_X, feature_X)
        # match_f = em.get_features(feature_X, feature_X, atypes1, atypes2, match_c, match_t, match_s)
        # match_f = match_f.sample(30).reset_index(drop=True)
        match_f=pd.DataFrame(columns=['feature_name', 'left_attribute', 'right_attribute',
       'left_attr_tokenizer', 'right_attr_tokenizer', 'simfunction',
       'function', 'function_source', 'is_auto_generated'])

        # # load feature table
        old_feature_table=pd.read_csv("model/feature_table_"+nstr[xidx]+".csv",index_col=0)
        old_feature_table=old_feature_table.fillna("None")
        # old_features=old_features['feature_name'].tolist()
        with open("model/old_features_"+nstr[xidx]+".data","rb") as f:
            raw_attrs=f.read()
        old_features = pickle.loads(raw_attrs)
        for i,row in old_feature_table[:].iterrows():
            if row['feature_name'] in match_f['feature_name'].to_list():
                continue
            if row['left_attr_tokenizer'] == "None" or row['left_attr_tokenizer'] is None:
                one="{}((ltuple.{}), (rtuple.{}))".format(row['simfunction'],row['left_attribute'],row['right_attribute'])
                r=em.get_feature_fn(one,match_t,match_s)
                em.add_feature(match_f,row['feature_name'],r)
            else:
                one="{}({}(ltuple.{}), {}(rtuple.{}))".format(row['simfunction'],row['left_attr_tokenizer'],row['left_attribute'],row['right_attr_tokenizer'],row['right_attribute'])
                r=em.get_feature_fn(one,match_t,match_s)
                em.add_feature(match_f,row['feature_name'],r)
        feature_table=match_f[match_f['feature_name'].isin(old_features)].reset_index(drop=True)
        feature_table=feature_table.set_index('feature_name')
        feature_table=feature_table.reindex(index=old_feature_table['feature_name'])
        feature_table=feature_table.reset_index()
        feature_table[['left_attribute','right_attribute','function_source',"left_attr_tokenizer","right_attr_tokenizer","simfunction"]]=old_feature_table[['left_attribute','right_attribute','function_source',"left_attr_tokenizer","right_attr_tokenizer","simfunction"]]
        assert(len(feature_table)==len(old_features))
        # load model
        rf=joblib.load("model/random_forest_"+nstr[xidx]+".joblib")

        brand_cluster=[]
        for bb in brand_list:
            brand_cluster.append([])
        for i,item in X.iterrows():
            if(item['brand_refine']!=""):
                brand_cluster[brand_list.index(item['brand_refine'])].append(i)
        brand_cluster=[X.loc[i] for i in brand_cluster]

        def applyMatch(merge_data,raw_X):
            if(len(merge_data)==0):
                return None
            merge_data=merge_data.merge(raw_X,left_on=['ltable_instance_id'],right_on=['instance_id'])
            merge_data=merge_data.merge(raw_X,left_on=['rtable_instance_id'],right_on=['instance_id'])
            merge_data=merge_data[merge_data['ltable_instance_id']!=merge_data['rtable_instance_id']]
            em.set_key(merge_data,'_id')
            em.set_fk_ltable(merge_data,'ltable_instance_id')
            em.set_fk_rtable(merge_data,'rtable_instance_id')
            em.set_ltable(merge_data,raw_X)
            em.set_rtable(merge_data,raw_X)
            em.set_key(raw_X,'instance_id')

            attrs_from_table = [one for one in merge_data.columns if one not in ['ltable_instance_id','rtable_instance_id','_id']]
            L = em.extract_feature_vecs(merge_data, feature_table=feature_table,
                                    attrs_before= attrs_from_table,
                                    show_progress=True, n_jobs=-1)
            em.set_key(L,'_id')
            attrs_to_be_excluded = []
            attrs_to_be_excluded.extend(['_id', 'ltable_instance_id','rtable_instance_id'])
            attrs_to_be_excluded.extend(attrs_from_table)
            output_data = rf.predict(table=L, exclude_attrs=attrs_to_be_excluded,                          
                        append=True, target_attr='predicted', inplace=False,probs_attr='probs',return_probs=True)
            for i,item in output_data[output_data['model_refine_x']=='8460p'].iterrows():
                if(item['ram_refine_x']=='6gb' and item['ram_refine_y']!='6gb'):
                    output_data.loc[i,"predicted"]=0
                elif(item['ram_refine_y']=='6gb' and item['ram_refine_x']!='6gb'):
                    output_data.loc[i,"predicted"]=0
            for i,item in output_data[output_data['model_refine_y']=='8460p'].iterrows():
                if(item['ram_refine_x']=='6gb' and item['ram_refine_y']!='6gb'):
                    output_data.loc[i,"predicted"]=0
                elif(item['ram_refine_y']=='6gb' and item['ram_refine_x']!='6gb'):
                    output_data.loc[i,"predicted"]=0
            for i,item in output_data[output_data['model_refine_x']=='2325'].iterrows():
                if(item['ram_refine_x']=='4gb' and item['ram_refine_y']=='8gb'):
                    output_data.loc[i,"predicted"]=0
                elif(item['ram_refine_x']=='8gb' and item['ram_refine_y']=='4gb'):
                    output_data.loc[i,"predicted"]=0
            for i,item in output_data[output_data['model_refine_y']=='2325'].iterrows():
                if(item['ram_refine_x']=='4gb' and item['ram_refine_y']=='8gb'):
                    output_data.loc[i,"predicted"]=0
                elif(item['ram_refine_x']=='8gb' and item['ram_refine_y']=='4gb'):
                    output_data.loc[i,"predicted"]=0
            output_data =  output_data[output_data['probs']>0.45]
            return output_data
        merge_cluster=[applyMerge(item) for item in brand_cluster]
        match_cluster=[applyMatch(item,X) for item in merge_cluster]
        result_df=pd.concat(match_cluster)
        result_df=result_df[result_df['ltable_instance_id']!=result_df['rtable_instance_id']]
        df_result=result_df
        df_result=df_result[['ltable_instance_id','rtable_instance_id']]
        df_result.columns=['left_instance_id','right_instance_id']
        df_result=deleteUncertainMatches(df_result)
        df_result=transitivelyCloseResult(df_result)


    if os.path.isfile("output.csv"):
        old_output=pd.read_csv("output.csv")
        if(len(old_output)<2):
            df_result.to_csv("output.csv",index=False)
        else:
            df_result=pd.concat([old_output,df_result])
            df_result.to_csv("output.csv",index=False)
    else:
        df_result.to_csv("output.csv",index=False)
