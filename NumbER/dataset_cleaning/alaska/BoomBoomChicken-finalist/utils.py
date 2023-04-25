import py_entitymatching as em
import os
import pandas as pd
import numpy as np
import copy
from collections import defaultdict
import itertools



def applyMerge(X_data):
    id_list=X_data['instance_id'].to_list()
    merge_data=pd.DataFrame(list(itertools.product(id_list, id_list)), columns=['ltable_instance_id', 'rtable_instance_id'])
    # X_data=X_data.reset_index()
    # em.set_key(X_data,'instance_id')
    # merge_data=rb.block_tables(X_data,X_data)
    merge_data=merge_data[merge_data["ltable_instance_id"]!=merge_data["rtable_instance_id"]]
    merge_data['check_string']=""
    merge_data['check_string'] = merge_data.apply(lambda row: ''.join(sorted([row['ltable_instance_id'], row['rtable_instance_id']])), axis=1)
    merge_data.drop_duplicates('check_string',inplace=True)
    merge_data=merge_data.drop('check_string',axis=1)
    merge_data.reset_index(drop=True)
    merge_data['_id']=merge_data.index
    return merge_data

def transferOneTree(tree):
    n_nodes_ = tree.node_count
    children_left_ = tree.children_left
    children_right_ = tree.children_right
    feature_ = tree.feature
    threshold_ = tree.threshold
    value_=tree.value
    queue=[]
    frule=Rule(0)
    queue.append(frule)
    visited=[]
    visited_true=[]
    visited_false=[]
    while len(queue)!=0:
        crule=queue.pop(0)
        croot=crule.idx
        cleft=children_left_[croot]
        cright=children_right_[croot]
        if (cleft==-1 and cright==-1):
            class_idx = np.argmax(value_[croot])
            crule.value=value_[croot][0][class_idx]
            crule.type= True if class_idx==1 else False
#             print("finish one path!")
            visited.append(crule)
            if crule.type is True:
                visited_true.append(crule)
            else:
                visited_false.append(crule)
        else:
            lrule=copy.deepcopy(crule)
            rrule=copy.deepcopy(crule)
            lrule.add(feature_[croot],-1,threshold_[croot])
            rrule.add(feature_[croot],1,threshold_[croot])
            lrule.idx=cleft
            rrule.idx=cright
            queue.append(lrule)
            queue.append(rrule)

    #return visited
    return visited_true,visited_false
        
class Rule(object):
    def __init__(self,idx):
        self.length=0
        self.type=-1
        self.features=[]
        self.signs=[]   # sign = -1 means "<", 1 means">""
        self.thresholds=[]
        self.value=0
        self.idx=idx
        
    def add(self,feature,sign,threshold):
        self.features.append(feature)
        self.signs.append(sign)
        self.thresholds.append(threshold)
        self.length+=1

def printOneRule(r,tree_feautres):
    if r.type ==True:
        is_match="match"
    else:
        is_match="nomatch"
    print("A {} path:".format(is_match))
    for i in range(r.length):
        if(r.signs[i]==1):
            sign=">"
        else:
            sign="<="
        feature=tree_features[r.features[i]]
#         feature=r.features[i]
        print("\t({feature} {inequality} {threshold})".format(feature=feature,inequality=sign,threshold=r.thresholds[i]))
        
def ruleToBlocker(r,tree_features):
    assert r.type==False
    rbs=[]
    for i in range(r.length):
        if(r.signs[i]==1):
            sign=">"
        else:
            sign="<="
        feature=tree_features[r.features[i]]
        orb=feature+("(ltuple,rtuple) ")+sign+" "+ str(r.thresholds[i])
        rbs.append(orb)
    return rbs

class Instance(object):
    def __init__(self,_id):
        self._id=_id
        self.parent=self

def find(x):
    if x.parent._id==x._id:
        return x
    else:
        x.parent= find(x.parent)
        return x.parent

def union(x,y):
    xroot=find(x)._id
    yroot=find(y)._id
    if xroot != yroot:
        y.parent=x.parent
        
def createResult(arr,lcol,rcol):
    if(len(arr)<2):
        return 
    else:
        for i in range(1,len(arr)):
            lcol.append(arr[0])
            rcol.append(arr[i])
    return createResult(arr[1:],lcol,rcol)

def transitivelyCloseResult(result_df):
    id2Ins=defaultdict(Instance)
    for one in set(list(result_df['left_instance_id'])+list(result_df['right_instance_id'])):
        oIns=Instance(one)
        id2Ins[one]=oIns
    all_instance=[]
    for _,row in result_df.iterrows():
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
    lcol=[]
    rcol=[]
    for k, v in clusters.items():
        createResult(list(set(v)),lcol,rcol)
    df_result=pd.DataFrame({"left_instance_id":lcol,"right_instance_id":rcol})
    df_result['check_string']=df_result.apply(lambda row: ''.join(sorted([row['left_instance_id'], row['right_instance_id']])), axis=1)
    df_result=df_result.drop_duplicates('check_string')
    df_result=df_result.drop('check_string',axis=1)
    return df_result

def deleteUncertainMatches(df):
    match_instances=list(df['left_instance_id'].unique())+list(df['right_instance_id'].unique())
    match_instances=list(set(match_instances))
    match_instances.sort()
    mmatrix=np.zeros((len(match_instances),len(match_instances))).astype(int)
    for i,item in df.iterrows():
        idx1=match_instances.index(item['left_instance_id'])
        idx2=match_instances.index(item['right_instance_id'])
        mmatrix[idx1,idx2]=1
        mmatrix[idx2,idx1]=1
    for i in range(len(match_instances)):
        mmatrix[i][i]=1

    # new_mmatrix=mmatrix.copy()
    for _ in range(2):
        new_mmatrix=mmatrix
        for i in range(len(match_instances)):
            for j in np.where(mmatrix[i]==1)[0]:
                count_same=np.count_nonzero((mmatrix[i]) & (mmatrix[j]))
                count_diff=np.count_nonzero((mmatrix[i]) ^ (mmatrix[j]))
                count_ones=np.count_nonzero((mmatrix[i]) | (mmatrix[j]))
                min_one=min(sum(mmatrix[i]),sum(mmatrix[j]))
                if count_same<count_diff  :
                    new_mmatrix[i][j]=0
                    new_mmatrix[j][i]=0
    lcol=[]
    rcol=[]
    for i in range(len(match_instances)):
        for j in np.where(new_mmatrix[i]==1)[0]:
            if(j>i):
                lcol.append(match_instances[i])
                rcol.append(match_instances[j])
    res_df=pd.DataFrame({'left_instance_id':lcol,'right_instance_id':rcol})
    return res_df
