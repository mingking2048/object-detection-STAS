# CBNetV2: A Novel Composite Backbone Network Architecture for Object Detection

##安裝虛擬環境(建議參考:https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)
版本及cuda和gpu型號都有不同需下安裝指令，請查閱以上網址
以下是我的版本
-------------------------------------
NAME="Ubuntu"
VERSION="18.04.6 LTS (Bionic Beaver)"
NVIDIA-SMI 440.33.01
Driver Version: 440.33.01
CUDA Version: 10.2
GPU:RTX2080TI*4
------------------------------------

Step 1. Create a conda environment and activate it.

conda create --name openmmlab python=3.8 -y
conda activate openmmlab

Step 2. Install PyTorch following official instructions, e.g.

On GPU platforms:

conda install pytorch torchvision -c pytorch

On CPU platforms:

conda install pytorch torchvision cpuonly -c pytorch

Step 3

pip install -U openmim
mim install mmcv-full

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

----------------------------------------------------------------------------------------------------
Verify the installation

mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file result.jpg

You will see a new image result.jpg on your current folder, where bounding boxes are plotted on cars, benches, etc.
-------------------------------------

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html(!!!請參考以上網址找對應的版本)

Step 4 install Apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
----------------------------------環境強烈建議參考官網指令----------------------------------------------


Training

step1 下載pretrain weight
https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth.zip
放入pretrain_model_weight資料夾
             |- htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth


step2 將要training data、annotation、testing data放入對應資料夾

將訓練圖片放入data/OBJ_Train_Datasets/Train_Images資料夾
							|-00000000.jpg
							|-00000001.jpg
							|-00000002.jpg
							.
							.
							.

將Annotations放入data/OBJ_Train_Datasets/Train_Annotations資料夾
							|-00000000.xmls
							|-00000001.xmls
							|-00000002.xmls
							.
							.
							.

將test圖片放入data/OBJ_Train_Datasets/Test_Images資料夾
							|-Private_00000000.jpg
							|-Private_00000001.jpg
							|-Private_00000002.jpg
							.
							.
							.
step3 產生訓練pickle檔
在code目錄中

python STAS2pickle.py --sp 0.9  (sp為 train val比例) file is save on: ./data/OBJ_Train_Datasets/pkl

step4 開始訓練
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch  tools/train.py configs/my_config/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_adamw_20e_coco_base.py --seed 12060911 --gpus 1 --deterministic   --work-dir ./work_dirs/new_epoch7

step5 finetune
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch  tools/train.py configs/my_config/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_adamw_20e_coco_finetune.py --seed 12060911 --gpus 1 --deterministic   --work-dir ./work_dirs/new_epoch7_fine

--------------------------------------------------------------------------------------------------

產生result json檔


若您已經按照上面的步驟訓練完畢即可略過，否則請下載我的weight放入./work_dirs/new_epoch7_fine資料夾
														|-latest.pth


CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/my_config/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_adamw_20e_coco_finetune.py work_dirs/new_epoch7_fine/latest.pth --out result_test.json (--out 可以指定result的位置)




@article{liang2021cbnetv2,
  title={CBNetV2: A Composite Backbone Network Architecture for Object Detection}, 
  author={Tingting Liang and Xiaojie Chu and Yudong Liu and Yongtao Wang and Zhi Tang and Wei Chu and Jingdong Chen and Haibing Ling},
  journal={arXiv preprint arXiv:2107.00420},
  year={2021}
}











