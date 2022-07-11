
#2021 12 22
CUDA_VISIBLE_DEVICES="0,1,2,3" nohup python -u train_img.py --model ResNet50 --batch-size 256 --lr 0.05 --pretrained True --target 0.7 --lossfact 2 --distillation_momentum 0.9 --expname imagenet_res50_T0.7*2_M0.9_bs256_lr0.05_stagewisecos --cos_loss 0.1 >log/imagenet_res50_T0.7*2_M0.9_bs256_lr0.05_epoch150.train &

