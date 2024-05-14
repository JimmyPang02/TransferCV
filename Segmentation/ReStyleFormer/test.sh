# Obtained from: https://github.com/yurujaja/DGInStyle-SegModel

#!/bin/bash

# 对模型实验结果的评估

#2  s0 DAFormer 49.0 
#59 s0 DAFormer+gen 52.9 
#60 s1 DAFormer+gen+SHADE 53.6 
#1 s0 DAFormer+gen+PixMix 53.7 ✔
#DAFormer+gen+PixMix+Re 56.1

#     "/mnt/DGInStyle-SegModel/work_dirs/local-exp1/240415_0205_gengtaReVTAug2cs_dgdacs_fdthings_srconly_daformer_sepaspp_sl_mitb5_poly10warm_s0_224f2"

TEST_ROOTS=(
    "/mnt/DGInStyle-SegModel/work_dirs/local-exp2/240414_0923_gta2cs_dgdacs_fdthings_srconly_rcs001_daformer_sepaspp_sl_mitb5_poly10warm_s0_44a85"
    "/mnt/DGInStyle-SegModel/work_dirs/local-exp59/240413_1832_gengtaCAug2cs_dgdacs_fdthings_srconly_daformer_sepaspp_sl_mitb5_poly10warm_s0_6286c"
    "/mnt/DGInStyle-SegModel/work_dirs/local-exp60/240413_1803_gengtaCAug2cs_dgdacs_fdthings_srconly_shade_daformer_sepaspp_sl_mitb5_poly10warm_s1_64075"
    "/mnt/DGInStyle-SegModel/work_dirs/ReOutput/Re-checkpoint222"
    "/mnt/DGInStyle-SegModel/work_dirs/ReOutput/Re-checkpoint595959"
    "mnt/DGInStyle-SegModel/work_dirs/ReOutput/Re-checkpoint606060"
    "/mnt/DGInStyle-SegModel/work_dirs/ReOutput/Re-checkpoint60159_v2"
)
for TEST_ROOT in "${TEST_ROOTS[@]}"
do
    DIR_NAME=$(basename "$TEST_ROOT")
    CONFIG_FILE="${TEST_ROOT}/${DIR_NAME}.json"
    
    # 判断 CHECKPOINT_FILE 是否存在，不存在则使用 latest.pth
    CHECKPOINT_FILE="${TEST_ROOT}/iter_40000.pth"
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
    fi
    
    SHOW_DIR="${TEST_ROOT}/preds" 
    
    echo 'Config File:' $CONFIG_FILE
    echo 'Checkpoint File:' $CHECKPOINT_FILE
    echo 'Predictions Output Directory:' $SHOW_DIR

    python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --dataset Cityscapes --show-dir ${SHOW_DIR} --opacity 1.0
done

# 对重参数化模型结果的评估
# TEST_ROOT="/mnt/DGInStyle-SegModel/work_dirs/ReOutput/Re-checkpoint16059"
# echo "TEST_ROOT: ${TEST_ROOT}"
# DIR_NAME=$(basename "$TEST_ROOT")
# CONFIG_FILE="${TEST_ROOT}/${DIR_NAME}.json"
# #CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
# CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
    
# SHOW_DIR="${TEST_ROOT}/preds"
# echo 'Config File:' $CONFIG_FILE
# echo 'Checkpoint File:' $CHECKPOINT_FILE
# echo 'Predictions Output Directory:' $SHOW_DIR

# python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --dataset Cityscapes --show-dir ${SHOW_DIR}