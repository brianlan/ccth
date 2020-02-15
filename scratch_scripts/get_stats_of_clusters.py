from pathlib import Path
import os

sku_dir = Path("/tmp/clust-exp/1037551")

cnt_in_each_cluster = {clust: len(os.listdir(sku_dir / clust)) for clust in os.listdir(sku_dir)}
sorted_cnt = sorted(cnt_in_each_cluster.items(), key=lambda x: x[1], reverse=True)
print(sorted_cnt)


