<<<<<<< HEAD
python models/pspnet_train.py -train data/data_train.txt -lab data/data_label.txt --weights pspnet50_ade20k --num_epoch 2 --learning_rate 0.00001 -optimizer SGD >  train_log_aws1.txt 
=======
python models/pspnet_train.py -train data/data_train.txt -lab data/data_label.txt --weights pspnet50_ade20k --num_epoch 1 --learning_rate 0.00001 >  train_log.txt 
>>>>>>> 7c5f1b97e5452d9aa2c2ef76c0f854a8b0bf5941

aws ec2 stop-instances --instance-ids i-0c7ec9b8c34e4912a 
