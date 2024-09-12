import sys
para = sys.argv[1]
data_name = 'synonyms_COCO'

with open(f'./data_gen/train_data_original_{data_name}.txt', 'r') as file:
    lines = file.readlines()

#去重
lines = [line.strip() for line in lines]
lines = [line.lower() for line in lines]
lines = list(set(lines))


#获取所有的类别名
ct = [] 
with open(f"./data_gen/{data_name}.txt", "r", encoding='utf-8') as f:
    for ann in f.readlines():
        ann = ann.replace('\'','').strip('\n')[1:-1].split(',')
        for an in ann:
            ct.append(an)


for line in lines:
    line = line.lower()
    #处理teddybear：
    if 'eddy' in line:
        index = line.find('eddy')
        if index > 0 and (line[index - 1] != ' ' and line[index - 1] != 't' and line[index - 1] != 'T'):
            line = line[:index] + ' t' + line[index:]
    #处理potted plant：
    if 'otted' in line:
        index = line.find('otted')
        if index > 0 and (line[index - 1] != ' ' and line[index - 1] != 'p' and line[index - 1] != 'P'):
            line = line[:index] + ' p' + line[index:]
    #处理hair dryer：
    line = line.replace("hairdryer","hair dryer")
    #处理a和单词连在一起的情况：
    for word in ct:
        line = line.replace(f'a{word}', f'a {word}').replace(f'A{word}', f'A {word}').replace(f'and{word}', f'and {word}')
    #处理donut    
    line = line.replace("anut","a donut")
    #处理remote
    line = line.replace("remote","remote control").replace("remotes","remote control")
    #处理fire hydrant
    line = line.replace("firehydrant","fire hydrant")

    # 将处理后的内容写回文件
    if para=="1":
        with open(f'./data_gen/finetune_data_{data_name}.txt', 'a') as file:
            file.write(line+'\n')
    else:
        with open(f'./data_gen/pretrain_data_{data_name}.txt', 'a') as file:
            file.write(line+'\n')

