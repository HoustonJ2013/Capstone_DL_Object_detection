python src/metrics_acc_iou.py --List_predict results/baseline-resnet34_dilated8-c1_bilinear-epoch_5_rgb2/rgb_list --List_true ./data/ADE20K_object150_val.txt > logs/log_mit_rgb

python src/metrics_acc_iou.py --List_predict results/baseline-resnet34_dilated8-c1_bilinear-epoch_5_gray/gray_list --List_true ./data/ADE20K_object150_val.txt > logs/log_mit_gray
