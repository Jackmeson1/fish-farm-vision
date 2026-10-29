# Command lists as they ran

These are the exact command lists that produced the results, kept as they ran. Four Google Colab machines worked in
parallel; inside the files they are called fish1 to fish4 (`gpu1*.sh` to `gpu4*.sh` here). fish1 to fish3 had an
NVIDIA RTX PRO 6000 (96 GB), fish4 an A100. Some lists wait for another job to finish (`while pgrep ...`) because
jobs shared a GPU. `ras.sh`, `dead.sh`, `video.sh` and `df3.sh` are the dead-fish runs. All paths assume `/root/fish`.
