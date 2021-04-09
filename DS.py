import csv

crowd_set_path = "Source/TlkAgg2/crowd_labels.tsv"
#crowd_set_path = "Source/crowd_labels_test.tsv"

# Key - task id; Value - dictionary <worker_id, anw>
all_data = {}

with open(crowd_set_path, encoding='utf-8-sig') as f:
    txt = f.read()
    sp = txt.split('\n')
    sp.pop(len(sp) - 1)

    for str in sp:
        worker_id = str.split('\t')[0]
        task_id = str.split('\t')[1]
        anw = str.split('\t')[2]

        if anw == "":
            continue

        if task_id in all_data:
            task_info = all_data[task_id]
            task_info[worker_id] = anw
            all_data[task_id] = task_info
        else:
            all_data[task_id] = {worker_id: anw}


def GetMajorityVoting(data):
    mv = {}

    for task_id in data:
        answers = list(all_data[task_id].values())

        # Key - answers; Value - its probability
        answers_probabilities = {}
        answers_without_duplicates = list(dict.fromkeys(answers))

        for unique_anw in answers_without_duplicates:
            anw_prob = answers.count(unique_anw) / len(answers)
            answers_probabilities[unique_anw] = anw_prob

        mv[task_id] = answers_probabilities

    return mv


# Calculate majority voting probability
all_data_mv = GetMajorityVoting(all_data)


# Key - worker_id; Value - dictionary with all answers prob
def UpdateWorkerErrors(all_worker_anws, worker_actual_anw, others_votes_probability):
    if worker_actual_anw in all_worker_anws:

        for excepted_anw in others_votes_probability:
            if excepted_anw not in all_worker_anws[worker_actual_anw]:
                all_worker_anws[worker_actual_anw][excepted_anw] = 0

            all_worker_anws[worker_actual_anw][excepted_anw] += others_votes_probability[excepted_anw]
    else:
        all_worker_anws[worker_actual_anw] = others_votes_probability.copy()

    return all_worker_anws


def CalculateWorkerErrors(data_with_mv):
    workers_errors = {}

    for task_id in data_with_mv:
        all_task_anw = all_data[task_id].copy()

        for task_worker_id in all_task_anw:

            actual_anw = all_task_anw[task_worker_id]
            other_workers_probability_votes = data_with_mv[task_id]

            # If that task-worker already in list
            if task_worker_id in workers_errors:
                # Update error-table
                all_worker_answers = workers_errors[task_worker_id]
                workers_errors[task_worker_id] = UpdateWorkerErrors(all_worker_answers, actual_anw, other_workers_probability_votes)

            else:
                # Add new data in dictionary
                workers_errors[task_worker_id] = {actual_anw: other_workers_probability_votes.copy()}

    return workers_errors


anw = CalculateWorkerErrors(all_data_mv)


def Normalization(worker_err_table):

    for worker_id in worker_err_table:
        # Find sum
        all_worker_anw = worker_err_table[worker_id]

        sum_of_unique_excepted_anw = {}
        for actual_anw in all_worker_anw:
            for expected_anw in all_worker_anw[actual_anw]:
                value = all_worker_anw[actual_anw][expected_anw]

                if expected_anw in sum_of_unique_excepted_anw:
                    sum_of_unique_excepted_anw[expected_anw] += value
                else:
                    sum_of_unique_excepted_anw[expected_anw] = value

        # Norm
        for actual_anw in all_worker_anw:
            for expected_anw in all_worker_anw[actual_anw]:
                if sum_of_unique_excepted_anw[expected_anw] == 0:
                    all_worker_anw[actual_anw][expected_anw] = 0
                    continue

                all_worker_anw[actual_anw][expected_anw] /= sum_of_unique_excepted_anw[expected_anw]

    return worker_err_table


worker_errors = Normalization(anw)

# Continue of M-step; Calculate Prior
def GetPriors(mv):
    priors = {}
    for task_id in mv:
        for anw in mv[task_id]:
            if anw in priors:
                priors[anw] += mv[task_id][anw]
            else:
                priors[anw] = mv[task_id][anw]

    norm_priors = {}
    priors_sum = sum(list(priors.values()))
    for prior in priors:
        norm_priors[prior] = priors[prior] / priors_sum

    return norm_priors


priors = GetPriors(all_data_mv)

# E-step
def mvRecalculation(priors, worker_errors):
    recalculated_mv = {}

    for task_id in all_data:
        if task_id not in recalculated_mv:
            recalculated_mv[task_id] = priors.copy()

        for worker_id in all_data[task_id]:
            worker_anw = all_data[task_id][worker_id]

            worker_actual_anws = worker_errors[worker_id][worker_anw]

            for worker_act_anw in recalculated_mv[task_id]:
                value_to_multy = 0

                if worker_act_anw in worker_actual_anws:
                    value_to_multy = worker_actual_anws[worker_act_anw]

                recalculated_mv[task_id][worker_act_anw] *= value_to_multy

        # Normalization
        sum_rec_mv = sum(list(recalculated_mv[task_id].values()))

        for anw in recalculated_mv[task_id]:
            recalculated_mv[task_id][anw] /= sum_rec_mv

    return recalculated_mv


print("Iteration 1")
recalculated_mv = mvRecalculation(priors, worker_errors)

print("Iteration 2")
anw = CalculateWorkerErrors(recalculated_mv)
worker_errors = Normalization(anw)
new_priors = GetPriors(recalculated_mv)
new_mv = mvRecalculation(new_priors, worker_errors)


def GetFinalAnw(mv):
    final_anws = {}

    for task_id in mv:
        final_anw = -1
        bigger_prob = -1

        for anw in mv[task_id]:
            if mv[task_id][anw] > bigger_prob:
                bigger_prob = mv[task_id][anw]
                final_anw = anw

        final_anws[task_id] = final_anw

    return final_anws


final_answers = GetFinalAnw(new_mv)


with open('AggregatedRes.tsv', 'wt', newline='') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')

    for key in final_answers:
        tsv_writer.writerow([key, final_answers[key]])