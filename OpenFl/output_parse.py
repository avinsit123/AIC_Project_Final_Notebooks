from tabulate import tabulate

f = open("test_output", "r")

table = [["Round", "party 1", "party 2", "party 3", "party 4", "party 5", "final model"]]

cur_round = 0
cur_metrics = []
flag = False
for line in f.readlines():
    words = line.split()
    if( (len(words) > 4)):
        if(flag):
            cur_metrics.append(float(words[-1]))
            flag = False
        print(words)
        if(words[0] == "METRIC" or words[1] == "METRIC"):
            if(words[1] == "METRIC"):
                words = words[1:]

            round = int(''.join(ch for ch in words[2] if ch.isalnum()))
            if(round != cur_round):
                table.append(cur_metrics)
                cur_round = round
                #cur_metrics = [cur_round]
                cur_metrics = []
            #if(words[3] == 'collaborator' and words[4].isdigit() and words[-4] == 'aggregated_model_validation:'):
                #cur_metrics.append(float(words[-2]))
            if(words[3] == 'aggregator:' and words[4] == 'aggregated_model_validation'):
                flag = True

table.append(cur_metrics)
print(tabulate(table, headers="firstrow"))
