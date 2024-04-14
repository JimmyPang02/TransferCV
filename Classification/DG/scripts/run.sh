dataset='office-home' # office-home
gpu_ids=0
data_dir='/mnt/DGdataset/OfficeHome/' 
max_epoch=20
net='resnet18'
task='img_dg'
output='/mnt/DeepDG/result'

# experiment
test_algorithms=('ERM' 'Mixup' 'MMD')
test_envs=('0' '1' '2' '3')
for alg in "${test_algorithms[@]}"; do
    for env in "${test_envs[@]}"; do
        echo $alg
        echo $env
        output_dir="${output}/${alg}_TestEnv${env}"
        if [ ! -d "$output_dir" ]; then
            mkdir -p "$output_dir" # 如果目录不存在，创建它
        fi
        python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output_dir \
        --test_envs $env --dataset $dataset --algorithm $alg
    done
done

