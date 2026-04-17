python -m scripts.validate_dstm \
  --ckpt outputs/measurement_140m_two_stage/joint_best.pt \
  --measurement-file data/measurement/wv_h100_180_online_imputed.mat \
  --nwp-file data/nwp/data_grid_online.mat \
  --device mps \
  --batch-size 16 \
  --out-dir outputs/validation_online_ide


python -m scripts.adapt_online \
  --ckpt outputs/offline_140m_01/offline_best.pt \
  --measurement-file data/measurement/wv_h100_180_online_imputed.mat \
  --nwp-file data/nwp/data_grid_online.mat \
  --device mps \
  --local-steps 3 \
  --out-dir outputs/online_140m_01

python -m scripts.validate_online_model \
  --ckpt outputs/offline_140m_01/offline_best.pt \
  --measurement-file data/measurement/wv_h100_180_online_imputed.mat \
  --nwp-file data/nwp/data_grid_online.mat \
  --device mps \
  --history-len 16 \
  --local-steps 3 \
  --out-dir outputs/validation_offline_01

# mac
python -m scripts.validate_multistep \
  --ckpt /Users/felix/Projects/wind_ide/outputs/offline_140m_01/offline_best.pt \
  --measurement-file /Users/felix/Projects/wind_ide/data/measurement/wv_h100_180_online_imputed.mat \
  --nwp-file /Users/felix/Projects/wind_ide/data/nwp/data_grid_online.mat \
  --history-len 24 \
  --horizon 12 \
  --out-dir /Users/felix/Projects/wind_ide/outputs/validation_multistep_12
# palmetto
python -m scripts.validate_multistep \
  --ckpt ~/wind_ide/outputs/offline_140m_01/offline_best.pt \
  --measurement-file ~/wind_ide/data/measurement/wv_h100_180_online_imputed.mat \
  --nwp-file ~/wind_ide/data/nwp/data_grid_online.mat \
  --history-len 24 \
  --horizon 12 \
  --out-dir ~/wind_ide/outputs/validation_multistep_12