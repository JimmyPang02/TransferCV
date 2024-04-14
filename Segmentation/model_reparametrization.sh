#!/usr/bin/env bash
python ./model_reparameterization.py latest.pth \
  --checkpoints /mnt/DGInStyle-SegModel/work_dirs/local-exp59/240413_1832_gengtaCAug2cs_dgdacs_fdthings_srconly_daformer_sepaspp_sl_mitb5_poly10warm_s0_6286c/iter_40000.pth \
  /mnt/DGInStyle-SegModel/work_dirs/local-exp59/240413_1832_gengtaCAug2cs_dgdacs_fdthings_srconly_daformer_sepaspp_sl_mitb5_poly10warm_s1_c94ec/iter_40000.pth \
  /mnt/DGInStyle-SegModel/work_dirs/local-exp59/240413_1832_gengtaCAug2cs_dgdacs_fdthings_srconly_daformer_sepaspp_sl_mitb5_poly10warm_s2_c8439/iter_40000.pth \
  --cpu-only --weights-filter ".*backbone.*" # 正则匹配参数名中带有“backbone”的参数