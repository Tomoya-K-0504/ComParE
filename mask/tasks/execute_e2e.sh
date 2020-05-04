for model in cnn rnn cnn_rnn
do
  time python tasks/e2e_experiment.py --manifest-path lab/labels.csv --cuda --epochs 30 --model-type $model \
  --amp --tensorboard --return-prob --n-parallel 1 --batch-size 64 --expt-id e2e_$model
done
