my_name="Synthetic-IHR-NomDB"

OUTPUT_DIR='./output/Synthetic-IHR-NomDB'
DATA_PATH="../Dataset/Synthetic-IHR-NomDB/generated_augmented/train"
TOKENIZER_PATH="./tokenizer-weights/dall_e_tokenizer_weight/"
                                                                                                                 

# ============================ pretraining ============================
python tools/run_pretraining.py \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model cae_small_patch48x4_48x432_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
  --batch_size 128 --lr 1.5e-3 --warmup_epochs 1 --epochs 10 \
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