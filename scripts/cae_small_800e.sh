tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}

OUTPUT_DIR='./output/'$my_name
DATA_PATH="../Dataset/Synthetic-IHR-NomDB/generated_augmented/generated_augmented/train"
TOKENIZER_PATH="./tokenizer-weights/dall_e_tokenizer_weights/"

ADDRESS=ADDR_FOR_THIS_MACHINE                                                                                 
NNODES=4     
RANK=RANK_FOR_THIS_MACHINE                                                                                                                        

# ============================ pretraining ============================
OMP_NUM_THREADS=1 python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --nnodes=$NNODES \
  --node_rank=$RANK \
  --master_addr=$ADDRESS \
  --master_port=8899 \
  tools/run_pretraining.py \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model cae_small_patch48x4_48x432_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
  --batch_size 4096 --lr 1.5e-4 --warmup_epochs 0.5 --epochs 10 \
  --clip_grad 3.0 --layer_scale_init_value 0.1 \
  --imagenet_default_mean_and_std \
  --color_jitter 0 \
  --drop_path 0.1 \
  --sincos_pos_emb \
  --mask_generator random \
  --ratio_mask_patches 0.45 \
  --decoder_layer_scale_init_value 0.1 \
  --no_auto_resume \
  --save_ckpt_freq 100 \
  --exp_name $my_name \
  --regressor_depth 4 \
  --decoder_depth 4 \
  --align_loss_weight 0.05
  # --align_loss_weight 2 \
  # --num_mask_patches 98 \