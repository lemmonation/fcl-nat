PROBLEM=translate_ende_wmt32k
DATA_DIR=$HOME/data/$PROBLEM
USR_DIR=$HOME/nat_wmt14

MODEL=transformer_nat_cl_word
HPARAMS=transformer_nat_cl_base_v1
EXTRA_SETTING_STR=set0
EXTRA_HPARAMS="at_pretrain_steps=100000,start_nat_attention_bias_step=200000"

TRAIN_DIR=$HOME/nat_baseline/model/$PROBLEM/$EXTRA_SETTING_STR
mkdir -p $TRAIN_DIR

CODE_DIR=$HOME/tensor2tensor
bin_file=$CODE_DIR/bin

export PYTHONPATH=$CODE_DIR:$PYTHONPATH

python ${bin_file}/t2t-trainer \
  --t2t_usr_dir=${USR_DIR} \
  --data_dir=${DATA_DIR} \
  --problems=${PROBLEM} \
  --model=${MODEL} \
  --hparams_set=${HPARAMS} \
  --hparams=$EXTRA_HPARAMS \
  --output_dir=${TRAIN_DIR} \
  --keep_checkpoint_max=1000 \
  --worker_gpu=8 \
  --train_steps=2000000 \
  --save_checkpoints_secs=2000 \
  --schedule=train \
