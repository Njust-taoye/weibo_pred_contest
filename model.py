#! /usr/bin/python
# -*- coding: utf-8 -*-
## @Author: taoye01
## @File: feature.py
## @Created Time: Thu 27 Dec 2018 01:33:40 PM CST
## @Description:含NGram(5000),字数,曜日(onehot),set(字数)特征，0.216726726727
import pdb
import copy,os,sys,psutil
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import jieba
import sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Pool,Manager
import re
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import datetime
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import make_scorer


def _remove_noise(document):
    noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
    clean_text = re.sub(noise_pattern, "", document)
    symbol_patt = re.compile('[\[+, \]+,【+, 】+,。+,{+, }+, !+,！+,?+,？+,<+,>+,《+,》+,（+,）+,\(+,\)+]')
    clean_text = re.sub(symbol_patt, "", clean_text)
    return clean_text

def get_stop_words():
    stop_words = []
    with open('stopwords_cn.txt') as fp:
        for line in fp.readlines():
            stop_word = line.strip().decode('utf-8')
            stop_words.append(stop_word)
    return stop_words

def get_weekday(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday()

def onehot(labels, label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(label_class)] for j in range(len(labels))]) 
    return one_hot_label


def cal_score(real_Y, pred_Y):
    real_fc = real_Y[:, 0]
    real_cc = real_Y[:, 1]
    real_lc = real_Y[:, 2]
    pred_fc = pred_Y[:, 0]
    pred_cc = pred_Y[:, 1]
    pred_lc = pred_Y[:, 2]
    dev_f = np.abs(pred_fc - real_fc)/(real_fc + 5) 
    dev_c = np.abs(pred_cc - real_cc)/(real_cc + 3)
    dev_l = np.abs(pred_lc - real_lc)/(real_lc + 3)
    dev = 1 - 0.5*dev_f - 0.25*dev_c - 0.25*dev_l
    def func(x):
        if x - 0.8 > 0:
            return 1
        else:
            return 0
    dev = np.array([func(x) for x in dev])
    count = real_fc + real_cc + real_lc + 1
    def func2(x):
        if x > 100:
            return 100
        else:
            return x
    count = np.array([func2(x) for x in count])
    res = sum(count * dev)/ float(sum(count))
    if tune_mode:
        return -res
    return res

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def tune_model(train_X, train_Y):
    print ('获取内存占用率： '+(str)(psutil.virtual_memory().percent)+'%')
    tune_params = {'n_estimators': range(100, 500,100), 'max_features': ['sqrt', 'log2'], 'max_depth': range(100, 500, 100), \
            'min_samples_split': [2, 4, 6], 'min_samples_leaf': [2, 3,5]} 
    
    #tune_params = {'n_estimators': [500], 'max_features': ['sqrt'], 'max_depth': [100]}
    score = make_scorer(cal_score, greater_is_better=False)
    gsearch = GridSearchCV(estimator = RandomForestRegressor(oob_score=False, random_state=10, n_jobs=1), param_grid = tune_params,\
                                                            scoring=score, cv=5)
    gsearch.fit(train_X, np.array(train_Y))
    print gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
    print ("Best score: %0.3f" % gsearch.best_score_)
    print ("Best paramters set:")
    best_paramters = gsearch.best_estimator_.get_params()
    for param_name in sorted(tune_params.keys()):
        print("\t%s: %r" % (param_name, best_paramters[param_name]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--addWeekday", help="add weekday feature if True", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--add_NGram_fea", help="add content NGram fea if True", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--NGram_num", help="NGram feature num", type=int, default=5000)
    parser.add_argument("--tune_mode", help="tune_model if True", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--add_words_num", help="add content words num feature if True", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--add_set_words_num", help="add content set words num feature if True", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--add_usr_fea", help="add usr features if True", type=str2bool, nargs="?", const=True, default=True)


    args = parser.parse_args()
    print "是否增加weekday特征", args.addWeekday
    print "是否增加unigram，bigram特征", args.add_NGram_fea
    print "文本特征维度", args.NGram_num
    print "是否在调参模式", args.tune_mode
    print "是否增加文章字数特征", args.add_words_num
    print "是否增加set文章字数特征", args.add_set_words_num
    print "是否增加用户特征", args.add_usr_fea
    add_weekday_fea = args.addWeekday
    add_NGram_fea = args.add_NGram_fea
    NGram_fea_num = args.NGram_num
    tune_mode = args.tune_mode
    add_words_num_fea = args.add_words_num
    add_set_words_num_fea = args.add_set_words_num
    add_usr_fea = args.add_usr_fea
    filename = 'weibo_train_data.txt'
    df = pd.read_csv(filename, sep='\t', header=None, names=['uid', 'mid', 'time', 'fc', 'cc', 'lc', 'content'])
    df.dropna(inplace=True)
    #train_df = df[(df['time'] > "2015-03-10 00:00:00") & (df['time'] < "2015-07-01 00:00:00")]
    #eval_df = df[(df['time'] >= "2015-07-01 00:00:00")]
    train_df = df[(df['time'] >= "2015-06-25 00:00:00") & (df['time'] < "2015-07-01 00:00:00")]
    eval_df = df[(df['time'] >= "2015-07-01 00:00:00") & (df['time'] < "2015-07-03 00:00:00")]
    print "train and eval has splited!!!!"
    if add_weekday_fea:
        train_df['weekday'] = train_df['time'].apply(lambda x: get_weekday(x))
        train_dayhot = onehot(list(train_df['weekday']), 7)
        train_dayhot = pd.DataFrame(train_dayhot, columns=range(7))
        train_df.reset_index(drop=True, inplace=True)#两表合并时，两表最好都做如此操作
        train_dayhot.reset_index(drop=True, inplace=True)
        train_df = pd.concat([train_df, train_dayhot], axis=1)
        train_df.drop(['weekday'], axis=1, inplace=True)
        eval_df['weekday'] = eval_df['time'].apply(lambda x: get_weekday(x))
        eval_dayhot = onehot(list(eval_df['weekday']), 7)
        eval_dayhot = pd.DataFrame(eval_dayhot, columns=range(7))
        eval_dayhot.reset_index(drop=True, inplace=True)
        eval_df.reset_index(drop=True, inplace=True)
        eval_df = pd.concat([eval_df, eval_dayhot], axis=1)
        eval_df.drop(['weekday'], axis=1, inplace=True)
        print "weekday onehoted!!!" 
    data_list = [] 
    train_words_num = []
    train_set_words_num = [] 
    def func(i):
        line = train_df.loc[i]
        content = line['content'].replace("\n", " ")
        words_num = len(content.decode('utf-8'))
        words_set_num = len(set(content.decode('utf-8')))
        word_cut = jieba.cut(content, cut_all=False)
        word_list = list(word_cut)
        word_list = ' '.join(word_list)
        res = []
        res.append(word_list)
        res.append(str(words_num))
        res.append(str(words_set_num))
        return '\t'.join(res)
    if not os.path.exists('cuted_word.txt'):
        print "cuted_word.txt not exists !!!"
        rst = []
        pool = Pool(8)
        for i in range(len(train_df)):
            rst.append(pool.apply_async(func, args=(i,)))
        pool.close()
        pool.join()
        print "cuted !!!!!!!!"
        rst = [i.get() for i in rst]
        with open('cuted_word.txt', 'w') as fp:
            for i in rst:
                fp.write(i+'\n')
                content, words_num, set_words_num = i.strip().split('\t')
                data_list.append(content)
                train_words_num.append(int(words_num))
                train_set_words_num.append(int(set_words_num))
        fp.close()
    else:
        print "cuted_word.txt has exists!!!!!"
        with open('cuted_word.txt', 'r') as rp:
            line = rp.readline()
            while line:
                content, words_num, set_words_num = line.strip().split('\t')
                train_words_num.append(int(words_num))
                train_set_words_num.append(int(set_words_num))
                data_list.append(content)
                line = rp.readline()
        rp.close()

    
    vectorizer = CountVectorizer(min_df=1, max_features=NGram_fea_num, ngram_range=(1,2), analyzer = 'word', 
                                stop_words = get_stop_words())
    if add_NGram_fea:
        train_nGram = vectorizer.fit_transform(data_list).toarray()
        if not os.path.exists(str(NGram_fea_num)+"words.txt"):
                with open(str(NGram_fea_num)+"words.txt", 'w') as fp:
                    for word in vectorizer.get_feature_names():
                        fp.write(word+"\n")
        train_nGram = pd.DataFrame(train_nGram, columns=range(NGram_fea_num))
        train_nGram.reset_index(inplace=True, drop=True)
        train_df.reset_index(inplace=True, drop=True)
        train_df = pd.concat([train_df, train_nGram], axis=1)
        print "has added NGram feature !!!!!"
    if add_words_num_fea:
        train_words_num = pd.DataFrame(train_words_num, columns=['words_num'])
        train_words_num.reset_index(inplace=True, drop=True)
        train_df.reset_index(inplace=True, drop=True)
        train_df = pd.concat([train_df, train_words_num], axis=1)
        print "has added words_num features!!!!"
    if add_set_words_num_fea:
        train_set_words_num = pd.DataFrame(train_set_words_num, columns=['set_words_num'])
        train_set_words_num.reset_index(inplace=True, drop=True)
        train_df.reset_index(inplace=True, drop=True)
        train_df = pd.concat([train_df, train_set_words_num], axis=1)
        print "has added set_words_num features!!!!!"
    if add_usr_fea:
        usr_info = pd.read_csv('usr_tmp.csv', header=0, sep='\t')
        usr_info.drop(['total_fc','total_cc','total_lc','total_words_num','total_set_words_num'], axis=1, inplace=True)
        #usr_info = usr_info[['uid', 'usr_freq', 'mean_words_num', 'mean_set_words_num']]
        train_df = pd.merge(train_df, usr_info, how='left',on='uid', suffixes=('','_usr'))
        train_df.fillna(0,inplace=True)
    train_Y = train_df[['fc', 'cc', 'lc']]
    train_X = train_df.drop(['uid', 'mid', 'time', 'content', 'fc', 'cc', 'lc'], axis=1)
    tune_mode = args.tune_mode
    if not tune_mode:
        print "model established!!!!!"
        rf = RandomForestRegressor(oob_score=False, random_state=10, n_jobs=3)
        #rf = RandomForestRegressor(n_estimators=500, max_features='log', max_depth=100, \
        #                            min_samples_split=4, min_samples_leaf=2)
        
        print rf.fit(train_X, train_Y)
    if tune_mode:
        print "tune model start !!!!!"
        tune_model(train_X, train_Y)
    print "model fited!!!!!"
    if not tune_mode: 
        eval_words_num = [] 
        eval_set_words_num = []
        eval_data_list = [] 
        def func(i):
            line = eval_df.loc[i]
            content = line['content'].replace("\n", " ")
            words_num = len(content.decode('utf-8'))
            set_words_num = len(set(content.decode('utf-8')))
            word_cut = jieba.cut(content, cut_all=False)
            word_list = list(word_cut)
            word_list = ' '.join(word_list)
            res = [] 
            res.append(word_list)
            res.append(str(words_num))
            res.append(str(set_words_num))
            return '\t'.join(res)
        if not os.path.exists('eval_words.txt'):
            print "eval_words.txt not exists!!!!!"
            rst = []
            pool = Pool(8)
            for i in range(len(eval_df)):
                rst.append(pool.apply_async(func, args=(i,)))
            pool.close()
            pool.join()
            rst = [i.get() for i in rst]
            with open('eval_words.txt', 'w') as fp:
                for i in rst:
                    content, words_num, set_words_num = i.strip().split('\t')
                    eval_data_list.append(content)
                    eval_words_num.append(int(words_num))
                    eval_set_words_num.append(int(set_words_num))
                    fp.write(i+'\n')
                    #fp.write('\n')
            fp.close()
        else:
            print "eval_words has exists !!!!!" 
            with open('eval_words.txt', 'r') as erp:
                line = erp.readline()
                while line:
                    content, words_num,set_words_num = line.strip().split('\t')
                    eval_data_list.append(content)
                    eval_words_num.append(int(words_num))
                    eval_set_words_num.append(int(set_words_num))
                    line = erp.readline()
            erp.close()
        if add_NGram_fea:
            eval_nGram = vectorizer.transform(eval_data_list).toarray()
            eval_nGram = pd.DataFrame(eval_nGram, columns=range(NGram_fea_num))
            eval_df.reset_index(inplace=True, drop=True)
            eval_nGram.reset_index(inplace=True, drop=True)
            eval_df = pd.concat([eval_df, eval_nGram], axis=1)
        if add_words_num_fea:
            eval_words_num = pd.DataFrame(eval_words_num, columns=['words_num'])
            eval_df.reset_index(inplace=True, drop=True)
            eval_words_num.reset_index(inplace=True, drop=True)
            eval_df = pd.concat([eval_df, eval_words_num], axis=1)
        if add_set_words_num_fea:
            eval_set_words_num = pd.DataFrame(eval_set_words_num, columns=['set_words_num'])
            eval_set_words_num.reset_index(inplace=True, drop=True)
            eval_df.reset_index(inplace=True, drop=True)
            eval_df = pd.concat([eval_df, eval_set_words_num], axis=1)
        if add_usr_fea:
            usr_info = pd.read_csv('usr_tmp.csv', header=0, sep='\t')
            usr_info.drop(['total_fc','total_cc','total_lc','total_words_num','total_set_words_num'], axis=1, inplace=True)
            #usr_info = usr_info[['uid', 'usr_freq', 'mean_words_num', 'mean_set_words_num']]
            eval_df = pd.merge(eval_df, usr_info, how='left',on='uid', suffixes=('','_usr'))
            eval_df.fillna(0,inplace=True)

        eval_Y = np.array(eval_df[['fc', 'cc', 'lc']])
        eval_X = eval_df.drop(['uid', 'mid', 'time', 'content', 'fc', 'cc', 'lc'], axis=1)
        pred_Y = rf.predict(eval_X)
        print cal_score(eval_Y, pred_Y)

