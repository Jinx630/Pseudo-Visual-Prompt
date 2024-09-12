data_name = 'synonyms_COCO'

data = []
with open(f"./tmp_data_nuswide/generate_sentence_{data_name}.txt", "r", encoding='utf-8') as f:
    for ann in f.readlines():
        ann = ann.strip('\n')
        if len(ann) < 2:
            continue
        if ann[0] == ' ':
            ann = ann[1:]
        if ann[0].isdigit() and ann[1] == '.':
            ok = 1
            for x in ann: # Have Chinese characters
                if ord(x) >= 256:
                    ok = 0
                    break
            tmpdata = ann.split(' ')
            for x in tmpdata: # Wrong sentence
                if len(x) > 15:
                    ok = 0
                    break
            if len(tmpdata)>22:
                ok=0
            if len(tmpdata)<7:
                ok=0
            if ok:
                data.append(ann[2:])
count=0
with open(f'./data_gen/train_data_original_{data_name}.txt', mode='w', encoding='utf-8') as file:
    for x in data:
        count+=1
        file.write(x + '\n')
print(count)