import os
import json
import numpy as np
import random

def load_codes(target_path):
    with open(target_path, 'r', encoding='utf-8') as file:
        codes = set()
        for line in file:
            line_codes = line.strip().split(' ')
            codes.update(line_codes)
        return codes


def load_queries_with_codes(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        file = json.load(file)
        queries = {}
        for img in list(file.keys()):
            if 'split'  not in file[img].keys():
                file.pop(img)
        for img in file.keys():
            item = file[img]
            cap = item['caption']
            code = item['code']
            queries[img] = code
            
        return queries


def average_intersection_size(sets_list):
    if len(sets_list) < 2:
        return 0  

    total_size = 0
    count = 0

    for i in range(len(sets_list)):
        for j in range(i + 1, len(sets_list)):
            intersection = sum(c1 == c2 for c1, c2 in zip(sets_list[i], sets_list[j]))
            total_size += intersection
            count += 1

    return total_size / count if count else 0


def retrieval_by_inersection(code, queries, top_k=50):
    distances = []
    for img in queries.keys():
        query = queries[img]
        distance = set(code).intersection(set(query))
        distances.append((img, query,len(distance)))
    distances.sort(key=lambda x: (-x[2], x[0]))

    # Adjust ranking to handle ties correctly
    ranked_results = []
    current_rank = 1
    last_distance = None
    for idx, (img, query, distance) in enumerate(distances, start=1):
        if distance != last_distance:
            current_rank = idx
        ranked_results.append((img, query, distance, current_rank))
        last_distance = distance
    
    return ranked_results[:top_k]

def retrieval_by_overlap(code, queries, top_k=50):
    distances = []
    for img in queries.keys():
        query = queries[img]
        distance = sum(c1 == c2 for c1, c2 in zip(code, query))
        distances.append((img, query,distance))
    distances.sort(key=lambda x: (-x[2], x[0]))

    # Adjust ranking to handle ties correctly
    ranked_results = []
    current_rank = 1
    last_distance = None
    for idx, (img, query, distance) in enumerate(distances, start=1):
        if distance != last_distance:
            current_rank = idx
        ranked_results.append((img, query, distance, current_rank))
        last_distance = distance
    
    return ranked_results[:top_k]

if __name__ == '__main__':

    train_mode = 'code_imgcap_v11_1024_pseudo_v3'
    code_file = 'data/flickr/flickr_codes_imgcap_v11_1024.json'

    print(f'Analyzing {train_mode} dataset...')
    train_source_file = 'data/flickr/'+train_mode+'/train.source'
    train_target_file = 'data/flickr/'+train_mode+'/train.target'
    val_source_file = 'data/flickr/'+train_mode+'/val.source'
    val_target_file = 'data/flickr/'+train_mode+'/val.target'
    test_source_file = 'data/flickr/'+train_mode+'/test.source'
    test_target_file = 'data/flickr/'+train_mode+'/test.target'

    train_codes = load_codes(train_target_file)
    dev_codes = load_codes(val_target_file)
    test_codes = load_codes(test_target_file)

    train_codes_count = len(train_codes)
    dev_codes_count = len(dev_codes)
    test_codes_count = len(test_codes)
    unique_codes_count = len(train_codes.union(dev_codes))
    common_codes_count = len(train_codes.intersection(dev_codes))
    test_unique_codes_count = len((train_codes.union(test_codes)))
    test_common_codes_count = len((train_codes.intersection(test_codes)))

    print(f'Number of codes in train: {train_codes_count}')
    print(f'Number of codes in dev: {dev_codes_count}')
    print(f'Number of codes in test: {test_codes_count}')
    print(f'Number of common codes between train and dev: {common_codes_count}')
    print(f'Number of common codes between train and test: {test_common_codes_count}')


    queries = load_queries_with_codes(code_file)
    print(f'\nNumber of images: {len(queries)}')


    with open('data/flickr/img_caption.json', 'r', encoding='utf-8') as file:
        file = json.load(file)
    rec = 0
    for cap in file.keys():
        item = file[cap]
        imgs = [x[0] for x in item]
        codes = [queries[x] for x in imgs]
        inter = average_intersection_size(codes)
        if inter == 1:
            print(cap)
            print(imgs)


   