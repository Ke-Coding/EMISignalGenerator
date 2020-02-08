PYTHONPATH=${PYTHONPATH}:../../ \
srun \
--job-name ${PWD##*/} \
--mpi=pmi2 -p $1 -n1 --gres=gpu:2 \
--ntasks-per-node=2 \
--cpus-per-task=5 \
python -u -m mnist_test
