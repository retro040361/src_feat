import pandas as pd

dataset = "citeseer"
ratio = [0.1,0.2,0.3,0.4,0.5]
degree = [5,6,7,8,9]
zzz = []
for d in degree:
    for r in ratio:
        all_data = []
        title = f"avg_log/{dataset}_v2_r{r}_d{d}_0620.csv"
        df = pd.read_csv(title)
        avg_data = df.iloc[1]
        all_data.append(d)
        all_data.append(r)
        all_data.append(avg_data["test auc"])
        all_data.append(avg_data["hit@1"])
        all_data.append(avg_data["hit@3"])
        all_data.append(avg_data["hit@10"])
        all_data.append(avg_data["hit@20"])
        all_data.append(avg_data["hit@100"])
        zzz.append(all_data)

df = pd.DataFrame(zzz, columns=["degree","ratio","auc","hit@1","hit@3","hit@10","hit@20","hit@100"])
df.to_csv(f"{dataset}_v2_0620.csv")