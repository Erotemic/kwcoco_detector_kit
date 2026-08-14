Starting fresh where I had untracked experiments that sort of set things up,
but not in a stable way.

Manually downloaded: 


```

mkdir -p /data/users/jon.crall/fish/downloads

gdown \
    10tJsWRUJn_FMPwWKkW9S6-H6fOF3DKyB \
    -O /data/users/jon.crall/fish/downloads/VIAME-v0.22.7-rc2-Linux-64Bit.tar.gz


cd ~/code/kwcoco_detector_kit
bash projects/viame_fish_2026/scripts/setup_binaries.sh \
    /data/users/jon.crall/fish/downloads/VIAME-v0.22.7-rc2-Linux-64Bit.tar.gz
```
