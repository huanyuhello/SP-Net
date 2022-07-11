

# 2021 12 15
#  FCFCFCFCFC
# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --dataset cifar10 --model ResNet110 --target 0.8 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.0 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.0] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.0].train &

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --dataset cifar10 --model ResNet110 --target 0.8 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.5 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.5] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.5].train &

# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --dataset cifar10 --model ResNet110 --target 0.8 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.1 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.1] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.1].train &

# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --dataset cifar10 --model ResNet110 --target 0.8 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.2 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.2] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.2].train &

# POOL POOL POOL
# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --dataset cifar10 --model ResNet110 --target 0.8 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.5 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.5]_pool >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.5]_pool.train &

# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --dataset cifar10 --model ResNet110 --target 0.8 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.1 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.1]_pool >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.1]_pool.train &

# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --dataset cifar10 --model ResNet110 --target 0.8 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.2 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.2]_pool >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.2]_pool.train &


# Logit Logit Logit  GG

# 2021 12 16

# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.0 --expname cifar10_res110_[m0.6]_[cos_m0.0_l0.0] >log/cifar10_res110_[m0.6]_[cos_m0.0_l0.0].train &

# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.1 --expname cifar10_res110_[m0.6]_[cos_m0.0_l0.1] >log/cifar10_res110_[m0.6]_[cos_m0.0_l0.1].train &

# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.2 --expname cifar10_res110_[m0.6]_[cos_m0.0_l0.2] >log/cifar10_res110_[m0.6]_[cos_m0.0_l0.2].train &

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.5 --expname cifar10_res110_[m0.6]_[cos_m0.0_l0.5] >log/cifar10_res110_[m0.6]_[cos_m0.0_l0.5].train &

# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --target 0.01 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.0 --expname cifar10_res110_[t0.01_m0.6]_[cos_m0.0_l0.0] >log/cifar10_res110_[t0.01_m0.6]_[cos_m0.0_l0.0].train &

# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --target 0.6 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.0 --expname cifar10_res110_[t0.6_m0.6]_[cos_m0.0_l0.0] >log/cifar10_res110_[t0.6_m0.6]_[cos_m0.0_l0.0].train &

# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --target 0.4 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.0 --expname cifar10_res110_[t0.4_m0.6]_[cos_m0.0_l0.0] >log/cifar10_res110_[t0.4_m0.6]_[cos_m0.0_l0.0].train &

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --target 0.2 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.0 --expname cifar10_res110_[t0.2_m0.6]_[cos_m0.0_l0.0] >log/cifar10_res110_[t0.2_m0.6]_[cos_m0.0_l0.0].train &


# ###### faraway
# POOL POOL POOL
# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.1 --expname cifar10_res110_[t0.8_m0.6]_[Farcos_m0.0_l0.1] >log/cifar10_res110_[t0.8_m0.6]_[Farcos_m0.0_l0.1].train &

# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.5 --cos_loss 0.1 --expname cifar10_res110_[t0.8_m0.6]_[Farcos_m0.5_l0.1] >log/cifar10_res110_[t0.8_m0.6]_[Farcos_m0.5_l0.1].train &


# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.7 --cos_loss 0.1 --expname cifar10_res110_[t0.8_m0.6]_[Farcos_m0.7_l0.1] >log/cifar10_res110_[t0.8_m0.6]_[Farcos_m0.7_l0.1].train &

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.5 --cos_loss 0.1 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.5_l0.1] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.5_l0.1].train &


# 2021 12 18
# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 1.0 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l1.0] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l1.0].train &

# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.3 --cos_loss 1.0 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.3_l1.0] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.3_l1.0].train &

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin -0.5 --cos_loss 1.0 --expname cifar10_res110_[t0.8_m0.6]_[cos_m-0.5_l1.0] >log/cifar10_res110_[t0.8_m0.6]_[cos_m-0.5_l1.0].train &


# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.5 --cos_loss 1.0 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.5_l1.0] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.5_l1.0].train &

# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.6 --cos_loss 1.0 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.6_l1.0] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.6_l1.0].train &

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.8 --cos_loss 1.0 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.8_l1.0] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.8_l1.0].train &

# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.7 --cos_loss 1.0 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.7_l1.0] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.7_l1.0].train &

# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --target 0.6 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.6 --cos_loss 1.0 --expname cifar10_res110_[t0.6_m0.6]_[cos_m0.6_l1.0] >log/cifar10_res110_[t0.6_m0.6]_[cos_m0.6_l1.0].train &

# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --target 0.4 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.6 --cos_loss 1.0 --expname cifar10_res110_[t0.4_m0.6]_[cos_m0.6_l1.0] >log/cifar10_res110_[t0.4_m0.6]_[cos_m0.6_l1.0].train &

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --target 0.2 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.6 --cos_loss 1.0 --expname cifar10_res110_[t0.2_m0.6]_[cos_m0.6_l1.0] >log/cifar10_res110_[t0.2_m0.6]_[cos_m0.6_l1.0].train &



# 12 19
# reimplement

# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.001 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.001] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l0.001].train &

# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.5 --cos_loss 1e-4 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.5_l1e-4] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.5_l1e-4].train &

# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin -0.5 --cos_loss 1e-4 --expname cifar10_res110_[t0.8_m0.6]_[cos_m-0.5_l1e-4] >log/cifar10_res110_[t0.8_m0.6]_[cos_m-0.5_l1e-4].train &

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 1e-4 --expname cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l1e-4] >log/cifar10_res110_[t0.8_m0.6]_[cos_m0.0_l1e-4].train &


# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --target 0.4 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 1e-4 --expname cifar10_res110_[t0.4_m0.6]_[cos_m0.0_l1e-4] >log/cifar10_res110_[t0.4_m0.6]_[cos_m0.0_l1e-4].train &

# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --target 0.6 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.001 --expname cifar10_res110_[t0.6_m0.6]_[cos_m0.0_l0.001] >log/cifar10_res110_[t0.6_m0.6]_[cos_m0.0_l0.001].train &

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --target 0.2 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 1e-4 --expname cifar10_res110_[t0.2_m0.6]_[cos_m0.0_l1e-4] >log/cifar10_res110_[t0.2_m0.6]_[cos_m0.0_l1e-4].train &


# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --target 0.2 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 0.001 --expname cifar10_res110_[t0.2_m0.6]_[cos_m0.0_l0.001_rate] >log/cifar10_res110_[t0.2_m0.6]_[cos_m0.0_l0.001_rate].train &


# 12 20

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --target 0.6 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 1e-4 --expname cifar10_res110_[t0.6_m0.6]_[cos_m0.0_l1e-4] >log/cifar10_res110_[t0.6_m0.6]_[cos_m0.0_l1e-4].train &

# 12 21
# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.5 --cos_margin 0.0 --cos_loss 1e-4 --expname cifar10_res110_[t0.8_m0.5]_[cos_l1e-4] >log/cifar10_res110_[t0.8_m0.5]_[cos_l1e-4].train &

# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.7 --cos_margin 0.0 --cos_loss 1e-4 --expname cifar10_res110_[t0.8_m0.7]_[cos_l1e-4] >log/cifar10_res110_[t0.8_m0.7]_[cos_l1e-4].train &

# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --target 0.8 --dataset cifar100 --model ResNet110 --pretrained True --distillation_momentum 0.5 --cos_margin 0.0 --cos_loss 1e-4 --expname cifar100_res110_[t0.8_m0.5]_[cos_l1e-4] >log/cifar100_res110_[t0.8_m0.5]_[cos_l1e-4].train &

# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --target 0.8 --dataset cifar100 --model ResNet110 --pretrained True --distillation_momentum 0.6 --cos_margin 0.0 --cos_loss 1e-4 --expname cifar100_res110_[t0.8_m0.6]_[cos_l1e-4] >log/cifar100_res110_[t0.8_m0.6]_[cos_l1e-4].train &


# 12 22
# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.8 --expname 1222cifar10_res110_[t0.8_m0.8]_[cos_l1e-4] >log/1222cifar10_res110_[t0.8_m0.8]_[cos_l1e-4].train &
# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --target 0.8 --dataset cifar100 --model ResNet110 --pretrained True --distillation_momentum 0.8 --expname 1222cifar100_res110_[t0.8_m0.8]_[cos_l1e-4] >log/1222cifar100_res110_[t0.8_m0.8]_[cos_l1e-4].train &
# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --target 0.8 --dataset cifar100 --model ResNet110 --pretrained True --distillation_momentum 0.7 --expname 1222cifar100_res110_[t0.8_m0.7]_[cos_l1e-4] >log/1222cifar100_res110_[t0.8_m0.7]_[cos_l1e-4].train &

# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --target 0.8 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --expname 1222cifar10_res110_[t0.8_m0.6]_[cos_l1e-4]_01 >log/1222cifar10_res110_[t0.8_m0.6]_[cos_l1e-4]_01.train &

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --target 0.2 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --expname 1222cifar10_res110_[t0.2_m0.6]_[cos_l1e-4]_01 >log/1222cifar10_res110_[t0.2_m0.6]_[cos_l1e-4]_01.train &

# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --target 0.4 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --expname 1222cifar10_res110_[t0.4_m0.6]_[cos_l1e-4]_01 >log/1222cifar10_res110_[t0.4_m0.6]_[cos_l1e-4]_01.train &

# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --target 0.6 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --expname 1222cifar10_res110_[t0.6_m0.6]_[cos_l1e-4]_01 >log/1222cifar10_res110_[t0.6_m0.6]_[cos_l1e-4]_01.train &

# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --target 0.9 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --expname 1222cifar10_res110_[t0.9_m0.6]_[cos_l1e-4]_01 >log/1222cifar10_res110_[t0.9_m0.6]_[cos_l1e-4]_01.train &

# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --target 0.9 --dataset cifar10 --model ResNet110 --pretrained True --distillation_momentum 0.6 --expname 1222cifar10_res110_[t0.9_m0.6]_[cos_l1e-4]_02 >log/1222cifar10_res110_[t0.9_m0.6]_[cos_l1e-4]_02.train &

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --target 0.2 --dataset cifar100 --model ResNet110 --pretrained True --distillation_momentum 0.6 --expname 1222cifar100_res110_[t0.2_m0.6]_[cos_l1e-4]_01 >log/1222cifar100_res110_[t0.2_m0.6]_[cos_l1e-4]_01.train &