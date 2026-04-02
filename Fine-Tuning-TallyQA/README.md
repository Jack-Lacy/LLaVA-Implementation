The setting to finetune LLaVA using PowerShell after creating the syncthesized counting dataset can be seen below:

cd "$work\LLaVA"`
`
python -m llava.train.train ``
  --lora_enable True ``
  --lora_r 128 ``
  --lora_alpha 256 ``
  --lora_dropout 0.05 ``
  --mm_projector_lr 2e-5 ``
  --model_name_or_path lmsys/vicuna-7b-v1.5 ``
  --version v1 ``
  --data_path "$datajson" ``
  --image_folder "$vgroot" ``
  --vision_tower openai/clip-vit-large-patch14-336 ``
  --pretrain_mm_mlp_adapter "$pretrain" ``
  --mm_projector_type mlp2x_gelu ``
  --mm_vision_select_layer -2 ``
  --mm_use_im_start_end False ``
  --mm_use_im_patch_token False ``
  --image_aspect_ratio pad ``
  --group_by_modality_length True ``
  --bf16 True ``
  --output_dir "$out" ``
  --num_train_epochs 2 ``
  --per_device_train_batch_size 1 ``
  --gradient_accumulation_steps 16 ``
  --evaluation_strategy "no" ``
  --save_strategy "steps" ``
  --save_steps 200 ``
  --save_total_limit 2 ``
  --learning_rate 5e-5 ``
  --weight_decay 0.01 ``
  --warmup_ratio 0.05 ``
  --lr_scheduler_type "cosine" ``
  --logging_steps 20 ``
  --tf32 True ``
  --model_max_length 1024 ``
  --gradient_checkpointing True ``
  --dataloader_num_workers 2 ``
  --lazy_preprocess True
notepad "$env:USERPROFILE\llava_tallyqa_vg_win\evaluate_tallyqa_500.py"`

cd "$env:USERPROFILE\llava_tallyqa_vg_win"`
`
python evaluate_tallyqa_500.py
notepad "$env:USERPROFILE\llava_tallyqa_vg_win\evaluate_tallyqa_500.py"`
