python models/pspnet_pred.py -m pspnet50_ade20k --input_list psptest1_list --output_path results/ -f

python src/metrics_acc_iou.py --List_predict pred_list --List_true label_list 

#python src/concat_image.py --image_predict results/ADE_val_00000001.npy
