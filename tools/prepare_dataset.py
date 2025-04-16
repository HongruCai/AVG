import os
import json
import numpy as np
import random
import json
import string
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Dataset for language model training")

    parser.add_argument('--code_file', type=str, default='data/flickr/flickr_codes.json', help='code file')
    parser.add_argument('--dataset', type=str, default='flickr', help='dataset name')
    parser.add_argument('--output_dir', type=str, default='data/flickr/codes', help='output directory')
    parser.add_argument('--code_mark', type=str, default='c_', help='special code mark')
    parser.add_argument('--pseudo_file', type=str, default=None, help='pseudo query file')
    
    return parser.parse_args()


def write_data(source_file, target_file, caption, codes, code_mark):
    source_file.write(caption.replace('\n','') + '\n')
    for i,code in enumerate(codes):
        target_file.write(f"{code_mark}{code} ")
    target_file.write('\n')


def process_items(data, train_source_file, train_target_file, val_source_file, val_target_file, test_source_file, test_target_file, code_mark, psedu_file=None):
    rec_count = {'train': 0, 'val': 0, 'test': 0}

    for item in data.values():
        if 'split' not in item.keys():
            continue
        split = item['split']
        captions = item['caption']
        reshaped_code = np.array(item['code']).reshape(-1)

        source_file, target_file = {
            'train': (train_source_file, train_target_file),
            'val': (val_source_file, val_target_file),
            'test': (test_source_file, test_target_file)
        }[split]

        for caption in captions:
            write_data(source_file, target_file, caption, reshaped_code, code_mark)
            rec_count[split] += 1

    if psedu_file is not None:
        print('adding pseudo query')
        with open(psedu_file, 'r') as f:
            pseudo = json.load(f)
        for item in pseudo:
            img = item['image_id']  
            reshaped_code = np.array(data[img]['code']).reshape(-1)
            for cap in item['caption']:
                write_data(train_source_file, train_target_file, cap, reshaped_code, code_mark)
                rec_count['train'] += 1

    return rec_count


if __name__ == '__main__':
    
    args = parse_args()
    json_file_path = args.code_file
    code_mark = args.code_mark
    output_dir_name = args.output_dir
    psedu_file = args.pseudo_file
    dataset = args.dataset

    with open(json_file_path, 'r') as file:
        data = json.load(file)
    os.makedirs('data/'+dataset+'/'+output_dir_name, exist_ok=True)
    shutil.copyfile(os.path.join(os.path.dirname(json_file_path), 'codebook_embedding.pt'), 'data/'+dataset+'/'+output_dir_name+'/codebook_embedding.pt')
    with open('data/'+dataset+'/'+output_dir_name+'/train.source', 'w') as train_source_file, \
        open('data/'+dataset+'/'+output_dir_name+'/train.target', 'w') as train_target_file, \
        open('data/'+dataset+'/'+output_dir_name+'/val.source', 'w') as val_source_file, \
        open('data/'+dataset+'/'+output_dir_name+'/val.target', 'w') as val_target_file, \
        open('data/'+dataset+'/'+output_dir_name+'/test.source', 'w') as test_source_file, \
        open('data/'+dataset+'/'+output_dir_name+'/test.target', 'w') as test_target_file:

        rec_count = process_items(data, train_source_file, train_target_file, val_source_file, 
                                  val_target_file, test_source_file, test_target_file, code_mark, psedu_file)

        print('number of pairs in train: ', rec_count['train'])
        print('number of pairs in val: ', rec_count['val'])
        print('number of pairs in test: ', rec_count['test'])
