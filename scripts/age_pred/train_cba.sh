CUDA_VISIBLE_DEVICES=0,3 ./distributed_train.sh 2 12004 ../data/AgePrediction --model rexnet_200  --sched cosine --remode pixel --batch-size 64 --lr 0.0001 --recount 3 --img-size 128 --aa v0 -j 4 --pretrained --loss-fn logit_cba --FPR 0.01 --output-feature --head-type ClassifierHead --channels-output 1 --num-classes 7 --class-map class_map.txt --num_sync 512 --sigma 0.5 --output ./output/age_pred/logit_cba_0.01
