import csv

SPLITS = "/scratch1/pachecog/ArgumentAnnotatedEssays-2.0/train-test-split.csv"

train = []; valid = []; test = []

with open(SPLITS, 'rU') as fp:
    readcsv = csv.reader(fp)
    for row in list(readcsv)[1:]:
        essay, fold = row[0].split(";")
        if fold == '"TRAIN"':
            train.append(essay)
        elif fold == '"TEST"':
            test.append(essay)

    valid_split = int(len(train) * 0.8)
    valid = train[valid_split:]
    train = train[:valid_split]

print(len(train), len(valid), len(test))

train_txt = open("train.txt", "w")
valid_txt = open("valid.txt", "w")
test_txt = open("test.txt", "w")

with open("all.txt") as fp:
    for line in fp:
        essay = line.split('/')[0]
        if essay in train:
            train_txt.write(line)
        elif essay in valid:
            valid_txt.write(line)
        elif essay in test:
            test_txt.write(line)
        else:
            print("some error")
            exit()
