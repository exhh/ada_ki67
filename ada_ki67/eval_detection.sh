gpu=1
testdata=IHC
traindata=IHC
model=frcn
iteration=best
result=learned_models

datadir=datasets
num_cls=1
eval_result_folder=experiments

load_model=${result}/${traindata}-${model}/${model}-${iteration}.pth

python eval_detection.py ${load_model} --model ${model} \
    --num_cls ${num_cls} --gpu ${gpu} \
    --datadir ${datadir} \
    --dataset ${testdata} \
    --eval_result_folder ${eval_result_folder}
