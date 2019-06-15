#encoding=utf-8
import argparse
import time
import os, codecs
import json
import random
ISOTIMEFORMAT='%Y-%m-%d %X'
parser = argparse.ArgumentParser(description='JF17K2rv_json.')
parser.add_argument('--data_dir', dest='data_dir', type=str, help="The data dir.", default='./data')
parser.add_argument('--data_name', dest='data_name', type=str, help="The data name", default='JF17K_version1')
parser.add_argument('--n_prefix', dest='n_prefix', type=str, help="prefix for n-ary files", default='n-ary_')
args = parser.parse_args()

def get_schema():
    schema = {}
    with open(args.data_dir+'/'+args.data_name+'/relation.txt', 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            schema[line[0]] = int(line[1])
    return schema

def write_json(schema, t_t):
    g = open(args.data_dir+'/'+args.data_name+'/'+args.n_prefix+t_t+'.json', 'w')
    with open(args.data_dir+'/'+args.data_name+'/'+t_t+'.txt', 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if t_t == 'train':
                begin = 1
            elif t_t == 'test':
                begin = 2
            rel = line[begin-1]
            if schema[rel] != (len(line)-begin):
                print(line, "schema:", schema[rel])
            n_dict = {}
            for i in range(begin, len(line)):  #no primary triple
                n_dict[rel+str(i-begin)] = line[i]
            n_dict['N'] = len(line)-begin
            json.dump(n_dict, g)
            g.write("\n")
    g.close()

if __name__ == '__main__':
    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
    schema = get_schema()
    g = open(args.data_dir+'/'+args.data_name+'/'+args.n_prefix+'valid.json', 'w')
    g.close()
    arr = ['train', 'test']
    for i in arr:
        write_json(schema, i)
    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
