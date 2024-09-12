import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline
from tqdm import tqdm
from itertools import combinations
import torch
import random

tokenizer = AutoTokenizer.from_pretrained("../pretrain_weigths/chatglm-6B", trust_remote_code=True)
model = AutoModel.from_pretrained("../pretrain_weigths/chatglm-6B", trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'

data_name = 'synonyms_COCO'

def main():
    ct = []
    with open(f"./data_gen/{data_name}.txt", "r", encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.replace('\'','').strip('\n')[1:-1].split(',')
            ct.append(ann[0])
    
    all_combinations = []
    for num_words in range(1, 4):  # 遍历从一个单词到三个单词的情况
        word_combinations = combinations(ct, num_words)
        all_combinations.extend(word_combinations)
    
    BATCH_SIZE = 16
    for i in tqdm(range(0,len(all_combinations),BATCH_SIZE), desc="BATCH_SIZE"):
        query_list = []
        for j in range(BATCH_SIZE):
            if i+j == len(all_combinations):
                break
            word_list=[word.strip() for word in all_combinations[i+j]]
            string = '\''+',\''.join(word_list)+'\''
            query = "Make an English short sentence to describe a photo as simple as possible! " + \
                    "Requirements: Generate 5 short English sentences! Each sentence should less than 15 words and include key words:" + \
                    string + \
                    "!\n"
            query_list.append(query)
        inputs = tokenizer(query_list, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, max_length=256)
        outputs = tokenizer.batch_decode(outputs)
        with open(f'./data_gen/generate_sentence_{data_name}.txt', mode='a', encoding='utf-8') as file:
            for _1 in range(BATCH_SIZE):
                file.write(outputs[_1] + '\n')

if __name__ == "__main__":
    main()
