import re
import pandas as pd
import os
cwd = os.getcwd()

if cwd.split('/')[-1] == "src":
    cwd=cwd[:-3]

# 讀取.log文件
date = "0623"
subfolder = "v2"
weight = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
degree = [5,6,7,8,9]
ratio = [0.1,0.2,0.3,0.4,0.5]
for w in weight:
    for d in degree:
        for r in ratio:
            title = f"cora_{subfolder}_r{r}_d{d}_w{w}"
            all_result = []
            for i in range(1,2):
                result = []
                with open(f'{cwd}/log/{date}/{subfolder}/{title}_{i}_loss_origin.log', 'r') as f:
                    log_data = f.read()

                # 使用正則表達式來抓取需要的資料
                pattern_epoch_roc_ap = r'best link prediction epoch = (\d+), Val_roc = ([\d.]+), val_ap = ([\d.]+), test_roc = ([\d.]+), test_ap = ([\d.]+)'
                pattern_epoch_hit = r'best hit@(\d+) epoch = (\d+), hit@(\d+) = ([\d.]+), val = ([\d.]+)'

                matches_epoch_roc_ap = re.findall(pattern_epoch_roc_ap, log_data)
                matches_epoch_hit = re.findall(pattern_epoch_hit, log_data)

                # 輸出結果
                for match in matches_epoch_roc_ap:
                    epoch, val_roc, val_ap, test_roc, test_ap = match
                    result.append(epoch)
                    result.append(test_roc)
                    # print(f"Link Prediction: Epoch {epoch}, Val_roc {val_roc}, val_ap {val_ap}, test_roc {test_roc}, test_ap {test_ap}")

                for match in matches_epoch_hit:
                    hit_number, epoch, _, hit_value, val = match
                    result.append(epoch)
                    result.append(hit_value)
                    result.append(val)
                    # print(f"Hit@{hit_number}: Epoch {epoch}, hit@{hit_number} {hit_value}, val {val}")
                all_result.append(result)

            df = pd.DataFrame(all_result,columns=["epoch","test auc","epoch","hit@1","auc","epoch","hit@3","auc","epoch","hit@10","auc","epoch","hit@20","auc","epoch","hit@50","auc","epoch","hit@100","auc"])
            df=df.astype(float)
            column_means = df.mean()
            df.loc[len(df)] = column_means

            df.to_csv(f"{cwd}/avg_log/{title}_{date}.csv")