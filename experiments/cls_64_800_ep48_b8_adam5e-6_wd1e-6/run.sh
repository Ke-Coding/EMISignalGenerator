EXP_DIR=$(pwd)
SIG_LEN=64

EP=48
BS=8
OP=adam
SCHE=con
AF=relu
LR=0.000005
WD=0.000001
DROP=0.2

DATA_DIR="${EXP_DIR}/data_${SIG_LEN}.txt"

if [ ! -f ${DATA_DIR} ]; then
  echo "generate data..."
  cd ../../emi_sig
  sh gen.sh $SIG_LEN 1200 1200 1200 1200 1200 ${DATA_DIR}
  cd ${EXP_DIR}
  echo "complete."
else
  echo "data file already exists."
fi

PYTHONPATH=${PYTHONPATH}:../../ \
srun \
--job-name ${PWD##*/} \
--mpi=pmi2 -p $1 -n1 --gres=gpu:2 \
--ntasks-per-node=2 \
--cpus-per-task=5 \
python -u -m cls_solver \
--log_dir=cls \
--input_size=${SIG_LEN} \
--num_classes=5 \
--epochs=${EP} \
--batch_size=${BS} \
--optm=${OP} \
--sche=${SCHE} \
--af_name=${AF} \
--lr=${LR} \
--wd=${WD} \
--nowd \
--dropout_p=${DROP} \
--tb_lg_freq=8 \
--val_freq=16 \
--data_path=${DATA_DIR} \
