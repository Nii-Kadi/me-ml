# -*- coding: utf-8 -*-
"""
Created on Mon Sep 01 09:42:51 2014

@author: 6008895
"""

# TODO think of a better package structure!
import text2

import codecs
import string
import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.stats import hmean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cossim


# _____________________________________________________________________________
# 0. Initialize document data
#    - read in documents data
#      format is id,"..."\n
#      we expect the ids to be in ascending order
#    - CSV of docids and publication synopses
#      for all documents consumed during target period
DOCS  = "data/docs_00.csv"
RANK  = 12

docids, data = [], []
with codecs.open(DOCS, encoding="utf-8") as f:
    for line in f:
        toks = string.split(line.strip(), sep=",", maxsplit=1)
        docids.append(toks[0])
        data.append(toks[1][1:-1])
docids = np.array(docids)                       # is this a good idea?

# 1. Compute document-term matrix
#    - sparse matrix of document and corresponding tf*idf weights
tfidf = TfidfVectorizer(stop_words='english', tokenizer=text2.MyTokenizer())
#dtm  = tfidf.fit_transform(data)
#corp = tfidf.inverse_transform(dtm)              # how long do we keep this???
tdm = tfidf.fit_transform(data)
corp = tfidf.inverse_transform(tdm)
# >>> get rid of corp???

terms = np.array(tfidf.get_feature_names())
# dict mapping docid to collection of term ids
corpus = {}
for i,d in enumerate(corp):
    ids = []
    for t in d:
        ids.append(np.where(terms==t))
    corpus[docids[i]] = np.array(ids).flatten()
    
tdm = tdm.transpose()
# _____________________________________________________________________________


# _____________________________________________________________________________
# 2. Compute SVD on dtm
class LSIModel():
    
    def __init__(self, A, rank=0):
        """Calculates the singular value decomposition for matrix A.
        
        Parameters
        ----------
        A : sparse matrix
        
        Returns
        -------
        self : object
        
        """
        _u, _s, _v = np.linalg.svd(A, full_matrices=0)
        
        self.rank = rank

        self.U  = _u[:,:self.rank].copy()
        self.S  = _s[:self.rank].copy()
        self.SI = np.matrix(np.diag(self.S)).getI()
        self.VT = _v[:self.rank,:].copy()
        
        self._var = [ e/(_s**2).sum() for e in (_s**2).cumsum() ][self.rank-1]
        
    def var_explained(self):
        return self._var   
        
    def query(self, qT):
        q_hat = np.dot(qT, self.U).dot(self.SI)
        res = cossim(q_hat, self.VT.transpose()).flatten()
        return res
    
    # Fold in a new text document into our LSI database!
    def foldin_item(self, itemid, text):
        global users_items, uu_enh, ii_enh, docids
        # TODO check if itemid already exists???       
        
        # first, deal with folding in the new item!
        ttdm    = tfidf.transform([text])
        dT      = ttdm.todense()
        d_hat   = np.dot(dT, self.U).dot(self.SI)
        self.VT = np.hstack((lsi.VT, d_hat.T))
        
        # next, calculate this document's similarity vs. each user's reading
        # history
        # insert similarity score into users_items sparse matrix IFF exceed thr
        # create virtual doc
        A = sps.lil_matrix(users_items)    
        A._shape = (A.shape[0], A.shape[1]+1)
        for user in userids:
            ui = np.where(userids == user)[0].item()
            dd = users_items.getrow(ui).nonzero()[1]   # <<< including similar,
                                                       #     but unread items!
            tids = []
            for did in docids[dd]:
                tids.extend(terms[corpus[did]])
            vd = " ".join(tids)
#            tt = []
#            for d in dd:
#                tt.append()
#            vd = " ".join( np.concatenate([corp[e] for e in dd]) )
            
            tmp = tfidf.transform([ vd ])
            qT = tmp.todense()
            q_hat = np.dot(qT, self.U).dot(self.SI)   
            s = cossim(q_hat, d_hat)
            # >>> threshold also used when folding in new items <<<
            if s > 0.2:
                A[ui,-1] = s
        
        # TODO is this really the right thing to do???                    !!!!!
        # this here is O(n^2)                                             !!!!!
        users_items = A.tocsr()
        uu_enh = cossim(users_items)
        ii_enh = cossim(users_items.transpose())
        
        docids = np.append(docids, itemid)
        corpus[itemid] = ttdm.indices
        
        print "Docids: ", docids
        print corpus
        
# _____________________________________________________________________________


# _____________________________________________________________________________
# 3. Next, read in the users-items reading history data, 
#    creating an initial, raw sparse matrix of 1's where a user has read a
#    certain item
#    - userids in ascending order
DATA   = "data/users_items_00.csv"
userids = []
uptr, iptr = [], []
with codecs.open(DATA, encoding='utf-8') as f:
    for line in f:
        u,i = string.split(line.strip(), sep=",", maxsplit=1)
        if u not in userids:
            userids.append(u)
        uptr.append( userids.index(u) )
        #iptr.append( docids.index(i) )
        iptr.append( np.where(docids == i)[0].item() )

userids = np.array(userids)                       # is this a good idea?
users_items = sps.csr_matrix((np.ones(len(iptr)), 
                             (uptr,iptr)), 
                             shape=(len(userids), len(docids)))  

users_items_raw = users_items.copy()          
        
lsi = LSIModel(tdm.todense(), rank=RANK)
print lsi.var_explained()
# _____________________________________________________________________________


# _____________________________________________________________________________
# 4. Enhance original users-items sparse matrix
# impute values (where possible) for each user
# by taking a virtual document containing all the terms of all documents
# read by a user, and then computing document similarity of that virtual
# document vis-a-vis the document-term tfidf matrix
# ...perhaps use an arbitrary threshold?
tmpu, tmpi, tmpd = [], [], []
users_terms = {}           # updatable container of users and significant terms
for i, id in enumerate(userids):
    # create virtual doc
    d  = users_items.getrow(i).nonzero()[1]

    tids = []
    for did in docids[d]:
        tids.extend(terms[corpus[did]])
    vd = " ".join(tids)
    
    tmp = tfidf.transform([ vd ])
    ti  = np.argsort(-tmp.data)
    users_terms[id] = zip(tmp.indices[ti], tmp.data[ti])
    
    qT = tmp.todense()
    res = lsi.query(qT)

    # don't include values for docs already read!
    res[d] = 0
    # TODO parameterize this arbitrary threshold?
    # >>> threshold also used when folding in new items <<<
    ti   = np.where(res > 0.20)[0]
    tmpi.extend(ti)
    tmpu.extend(np.tile(i, ti.size))
    tmpd.extend(res[ti])
    
tmpm = sps.csr_matrix((tmpd, (tmpu,tmpi)), shape=users_items.shape)    
users_items = users_items + tmpm

# User-user similarity matrix, enhanced"
uu_enh = cossim(users_items)
uu_raw = cossim(users_items_raw)

# Item-item similarity matrix, enhanced"
ii_enh = cossim(users_items.transpose())
ii_raw = cossim(users_items_raw.transpose())


def __get_user_row(user):
    uid = np.where(userids == user)[0]
    if len(uid) > 0:
        uid = uid.item()
        return users_items.getrow(uid)
    else:
        return None


# _____________________________________________________________________________
##
#                                                                Public API
#
"""
    recommend_by_item
    
    item - itemid (string)
    
    **kwargs -
      - item_sim_min_thr: minimum level of similarity criteria to use when
                   selecting items
      - max_items: maximum number of recommendations to return     
"""
def recommend_by_item(item, **kwargs):
    global ii_enh, docids
    
    maxn = kwargs['max_items'] if kwargs.has_key('max_items') else 3
    min_thr = kwargs['item_sim_min_thr'] if kwargs.has_key('item_sim_min_thr') else 0.2
    hits = np.where(docids == item)[0]
    ids, val = [], []
    if hits.size>0:
        i = hits.item()
        r = ii_enh[i]
        r = [e for e in np.argsort(-r) if (e!=i and r[e]>=min_thr)]
        ids = docids[r[:maxn]]
        val = ii_enh[i][r[:maxn]]
    return np.array(zip(ids, val))
    

"""
    recommend_for_user
    
    user - userid (string)
    
    **kwargs -
      - user_sim_min_thr: minimum level of similarity criteria to use when
                   selecting users to look at
      - max_users: maximum number of users to look at when looking at their
                   reading history in order to make suggestions
      - item_sim_min_thr: minimum level of similarity criteria to use when
                   selecting items from other users' histories
      - max_items: maximum number of recommendations to return                 
"""
def recommend_for_user(user, **kwargs):
    global uu_enh, userids
    docs, scores = [], []
    
    item_sim_min_thr = kwargs['item_sim_min_thr'] if kwargs.has_key('item_sim_min_thr') else 0.2
    item_maxn = kwargs['max_items'] if kwargs.has_key('max_items') else 3

    # first, locate top N similar users...
    res = similar_users(user, **kwargs)
    
    if res.size > 0:
        user_row = __get_user_row(user)
        ri = np.where(user_row.data == 1.0)
        user_read = user_row.indices[ri]
        
        sim_hists = [ __get_user_row(e).todense().tolist()[0] for e in res[:,0] ]
        sim_hists = sps.csr_matrix(sim_hists)

        print "For ", user
        print user_row
        
        print "similar users: "
        print res
        print sim_hists        
        
        for i, d in enumerate(docids):
            if sim_hists.getcol(i).size > 0 and i not in user_read:
                y_hat = hmean(sim_hists.getcol(i).data)
                y     = user_row[0,i]
                s     = 1.0 - np.abs(y_hat-y)
                #print d, ": hmean=", y_hat, ", score=", s
                if s >= item_sim_min_thr:
                    docs.append(d)
                    scores.append(s)

        docs   = np.array(docs)                
        scores = np.array(scores) 
        ids    = np.argsort(-scores)[0:item_maxn+1]
        docs   = docs[ids]
        scores = scores[ids]
        
    ret = np.array(zip(docs, scores)) 
    return ret


def similar_users(user, **kwargs):
    global uu_enh, userids
    
    user_sim_min_thr = kwargs['user_sim_min_thr'] if kwargs.has_key('user_sim_min_thr') else 0.2
    user_maxn = kwargs['max_users'] if kwargs.has_key('max_users') else 3
    
    # first, locate top N similar users...
    hits = np.where(userids == user)[0]
    ids, val = [], []
    if hits.size>0:
        i = hits.item()
        r = uu_enh[i]
        r = [e for e in np.argsort(-r) if (e!=i and r[e]>=user_sim_min_thr)]
        ids = userids[r[:user_maxn]]
        val = uu_enh[i][r[:user_maxn]]
    res = np.array(zip(ids, val))
    return res


def update_user_reading(user, item):
    # TODO isn't there some other way to do this right?
    global users_items, uu_enh, ii_enh
    
    ui = np.where(userids == user)[0]
    ii = np.where(docids  == item) [0]
    
    if ui.size == 0:
        raise Exception("User %s not found!" % user)
    if ii.size == 0:
        raise Exception("Item %s not found!" % item)

    ui = ui.item()
    ii = ii.item()    
    
    tmp = sps.lil_matrix(users_items)    
    tmp[ui,ii] = 1.0
    
    # TODO is this really the right thing to do???                        !!!!!
    # this here is O(n^2)                                                 !!!!!
    users_items = tmp.tocsr()
    uu_enh = cossim(users_items)
    ii_enh = cossim(users_items.transpose())
    
    # TODO is this really the right thing to do???                        !!!!!
    ur  = users_items.getrow(ui)
    ri  = ur.indices[np.where(ur.data == 1.0)]

    tids = []
    for did in docids[ri]:
        tids.extend(terms[corpus[did]])
    vd = " ".join(tids)
    #vd = " ".join( np.concatenate([corp[e] for e in d]) )   
    _t = tfidf.transform([ vd ])
    ti  = np.argsort(-_t.data)
    users_terms[user] = zip(_t.indices[ti], _t.data[ti])
   
    
def term_significance(user):
    # TODO pull out feature names from elsewhere!
    rankings = [ (terms[e[0]], e[1]) for e in users_terms[user] ]
    return np.array(rankings)
    
# _____________________________________________________________________________

print "          VALIDATION"
print "-------------------------------"

#print ">> Item-item recommendations <<"
#topNsimilar_docs  = get_topNsimilar(ii_enh, docids)
#topNsimilar_docs2 = get_topNsimilar(ii_raw, docids)
#dtm = tdm.transpose()
#ss, n = 0, 0
#for id in docids:
#    res = topNsimilar_docs(id)
#    n += len(res)
#    for r in res:
#        rid, y_h = r
#        i_d1 = np.where(docids == id)[0].item()
#        d1   = dtm[i_d1]
#        i_d2 = np.where(docids == rid)[0].item()
#        d2   = dtm[i_d2]
#        y    = cossim(d).item()
#        ss  += np.square(y_h - y)
#
#print "Recommendations via hybrid CF: ", n
#print "RMSE: ", np.sqrt(ss/n) 
#print 
#
#ss, n = 0, 0
#for id in docids:
#    res = topNsimilar_docs2(id)
#    n += len(res)
#    for r in res:
#        rid, y_h = r
#        i_d1 = np.where(docids == id)[0].item()
#        d1   = dtm[i_d1]
#        i_d2 = np.where(docids == rid)[0].item()
#        d2   = dtm[i_d2]
#        y    = cossim(d).item()
#        ss  += np.square(y_h - y)
#
#print "Recommendations via raw CF: ", n
#print "RMSE: ", np.sqrt(ss/n) 
#print

#print ">> User-user based recommendations <<"
##topNsimilar_users  = get_topNsimilar(uu_enh, userids)
##topNsimilar_users2 = __get_topNsimilar(uu_raw, userids)
#
#n = 0
#missing = []
#for user in userids:
#    res = recommend_for_user(user, max_users=4)
#    print user
#    user_hist = __get_user_row(user)
#    user_unread = np.where(user_hist.data < 1.0)
#    print "unread: "
#    print zip(docids[user_hist.indices[user_unread]], user_hist.data[user_unread])
#    print
#    print "recommendations: "
#    print res
#    print "______________________________"
#    print
#
#        
#print "User-based recommendations via hybrid CF: ", n
#print