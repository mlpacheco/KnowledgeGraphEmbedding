import json

chang_folds = json.load(open("chang_folds.json"))

for issue in ["abortion", "evolution", "gay_marriage", "gun_control"]:
    print(issue)
    entities = set([])
    classes = set([])
    for i in range(0, 5):
        dev_fold = (i + 1) % 5
        test_posts = chang_folds[issue][i]
        dev_posts = chang_folds[issue][dev_fold]
        train_posts = []
        for j in range(0, 5):
            if j != i and j != dev_fold:
                train_posts += chang_folds[issue][j]

        train_txt = open("{0}/f{1}/train.txt".format(issue, i), "w")
        valid_txt = open("{0}/f{1}/valid.txt".format(issue, i), "w")
        test_txt = open("{0}/f{1}/test.txt".format(issue, i), "w")

        n_edges = 0; n_out = 0

        with open("{0}/all.txt".format(issue)) as fp:
            for line in fp:
                n1, rel, n2 = line.strip().split('\t')

                if rel == "HasStance":
                    entities.add(n1)
                    classes.add(n2)
                    n1 = int(n1)
                    if n1 in train_posts:
                        train_txt.write(line)
                    elif n1 in dev_posts:
                        valid_txt.write(line)
                    elif n1 in test_posts:
                        test_txt.write(line)
                else:
                    entities.add(n1)
                    entities.add(n2)
                    n1 = int(n1); n2 = int(n2)
                    n_edges += 1
                    if n1 in train_posts and n2 in train_posts:
                        train_txt.write(line)
                    elif n1 in dev_posts and n2 in dev_posts:
                        valid_txt.write(line)
                    elif n1 in test_posts and n2 in test_posts:
                        test_txt.write(line)
                    else:
                        # Assing according to one node
                        if n1 in train_posts:
                            train_txt.write(line)
                        elif n1 in dev_posts:
                            valid_txt.write(line)
                        elif n1 in test_posts:
                            test_txt.write(line)

                        n_out += 1

        print(n_out, n_edges, (n_out*1.0)/n_edges)
        train_txt.close()
        valid_txt.close()
        test_txt.close()

    with open("{0}/entities.dict".format(issue), "w") as fp:
        classes = list(classes)
        classes.sort()
        for i, k in enumerate(classes):
            fp.write("{0}\t{1}\n".format(i, k))

        entities = list(entities)
        entities.sort()
        for i, ent in enumerate(entities):
            fp.write("{0}\t{1}\n".format(i + len(classes), ent))
