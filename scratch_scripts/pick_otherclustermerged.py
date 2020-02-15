from pathlib import Path
import os
import shutil
from tqdm import tqdm


src_dir = Path("/tmp/clust-subbrand-val-sampled")
dst_dir = Path("/tmp/clust-subbrand-val-sampled-otherclustersmerged")
dst_dir.mkdir(parents=True, exist_ok=True)

for dir_name in tqdm(os.listdir(src_dir)):
    if Path(src_dir / dir_name / "other-clusters-merged").exists():
        shutil.copytree(src_dir / dir_name / "other-clusters-merged", dst_dir / dir_name)
    else:
        if Path(src_dir / dir_name / "0").exists():
            shutil.copytree(src_dir / dir_name / "0", dst_dir / dir_name)
        if Path(src_dir / dir_name / "1").exists():
            [shutil.copy(p, dst_dir / dir_name) for p in (src_dir / dir_name / "1").glob("*")]
        if Path(src_dir / dir_name / "2").exists():
            [shutil.copy(p, dst_dir / dir_name) for p in (src_dir / dir_name / "2").glob("*")]
