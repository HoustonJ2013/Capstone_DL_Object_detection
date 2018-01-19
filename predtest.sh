<<<<<<< HEAD
python models/pspnet_pred.py -m pspnet50_ade20k --input_list psptest1_list -f --output_path  results/ --weights  pspade20k_epoch0_sgd_lrn5
=======
python models/pspnet_pred.py -m pspnet50_ade20k --input_list psptest1_list -f --output_path  results/ --weights pspade20k_epoch0_adam_lrn5
>>>>>>> 7c5f1b97e5452d9aa2c2ef76c0f854a8b0bf5941

python src/metrics_acc_iou.py --List_predict pred_list --List_true label_list 

#python src/concat_image.py --image_predict results/ADE_val_00000001.npy
