EXP_DIR=$(pwd)
PRO_DIR=$(dirname $(dirname "$PWD"))

SIG_LEN=64
SIG_N1=100
SIG_N2=100
SIG_N3=100
SIG_N4=100
SIG_N5=100

DS_DIR="${PRO_DIR}/emi_sig"
DS_FILE="${DS_DIR}/datasets/data_${SIG_LEN}_${SIG_N1}_${SIG_N2}_${SIG_N3}_${SIG_N4}_${SIG_N5}.txt"


if [[ ! -f ${DS_FILE} ]]; then
  echo "generate data..."
  cd ${DS_DIR}
  sh gen.sh ${SIG_LEN} ${SIG_N1} ${SIG_N2} ${SIG_N3} ${SIG_N4} ${SIG_N5} ${DS_FILE}
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
--cfg_dir=config.yaml \
--log_dir=$(date +cls_%Y%m%d_%H_%M_%S) \
--data_dir=${DS_FILE} \
--num_gpu=1 \
--seed=0 \
--input_size=${SIG_LEN} \
--num_classes=5 \
#--load_dir= \
