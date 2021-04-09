import csv

crowd_set_path = "Source/TlkAgg5/crowd_labels.tsv"

# Key - task; Value - list of anw
crowd_dic = {}

with open(crowd_set_path, encoding='utf-8-sig') as f:
    txt = f.read()
    sp = txt.split('\n')

    for str in sp:
        task_id = str.split('\t')[1]
        anw = str.split('\t')[2]

        if task_id in crowd_dic:
            anw_list = crowd_dic[task_id]
            anw_list.append(anw)
            crowd_dic[task_id] = anw_list
        else:
            crowd_dic[task_id] = [anw]

anw_list = {}
print(len(crowd_dic))

for key in crowd_dic:
    print(f"K: {key}; V: {crowd_dic[key]}")
    answers = crowd_dic[key]

    most_freq_anw = max(set(answers), key=answers.count)
    #print(f"Most: {most_freq_anw}; list: {answers}")

    anw_list[key] = most_freq_anw

with open('AggregatedRes.tsv', 'wt', newline='') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')

    for key in anw_list:
        tsv_writer.writerow([key, anw_list[key]])