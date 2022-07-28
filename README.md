Based on the work of [Chen Wang](https://github.com/j96w/DenseFusion) <br /> and [Dinh-Cuong Hoang](https://github.com/hoangcuongbk80/Object-RPE/tree/iliad/DenseFusion) <br />

## Requirements
* Python 3.5
* PyTorch 1.0
* torchvision 0.2.2.post3
* PIL
* scipy
* numpy
* pyyaml
* logging
* cffi
* matplotlib
* Cython
* CUDA 9.0/10.0

```bash
$ pip3 --no-cache-dir install numpy scipy pyyaml cffi pyyaml matplotlib Cython Pillow
$ pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl
$ pip3 install torchvision==0.2.2.post3
```

## Code Structure
* **datasets**
	* **datasets/ycb**
		* **datasets/ycb/dataset.py**: Data loader for YCB_Video dataset.
		* **datasets/ycb/dataset_config**
			* **datasets/ycb/dataset_config/classes.txt**: Object list of YCB_Video dataset.
			* **datasets/ycb/dataset_config/train_data_list.txt**: Training set of YCB_Video dataset.
			* **datasets/ycb/dataset_config/test_data_list.txt**: Testing set of YCB_Video dataset.
	* **datasets/linemod**
		* **datasets/linemod/dataset.py**: Data loader for LineMOD dataset.
		* **datasets/linemod/dataset_config**: 
			* **datasets/linemod/dataset_config/models_info.yml**: Object model info of LineMOD dataset.
* **replace_ycb_toolbox**: Replacement codes for the evaluation with [YCB_Video_toolbox](https://github.com/yuxng/YCB_Video_toolbox).
* **trained_models**
	* **trained_models/ycb**: Checkpoints of YCB_Video dataset.
	* **trained_models/linemod**: Checkpoints of LineMOD dataset.
* **lib**
	* **lib/loss.py**: Loss calculation for model.
	* **lib/loss_refiner.py**: Loss calculation for iterative refinement model.
	* **lib/transformations.py**: [Transformation Function Library](https://www.lfd.uci.edu/~gohlke/code/transformations.py.html).
    * **lib/network.py**: Network architecture.
    * **lib/extractors.py**: Encoder network architecture adapted from [pspnet-pytorch](https://github.com/Lextal/pspnet-pytorch).
    * **lib/pspnet.py**: Decoder network architecture.
    * **lib/utils.py**: Logger code.
    * **lib/knn/**: CUDA K-nearest neighbours library adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda).
* **tools**
	* **tools/_init_paths.py**: Add local path.
	* **tools/eval_ycb.py**: Evaluation code for YCB_Video dataset.
	* **tools/eval_linemod.py**: Evaluation code for LineMOD dataset.
	* **tools/train.py**: Training code for YCB_Video dataset and LineMOD dataset.
* **experiments**
	* **experiments/eval_result**
		* **experiments/eval_result/ycb**
			* **experiments/eval_result/ycb/EFN6D_wo_refine_result**: Evaluation result on YCB_Video dataset without refinement.
			* **experiments/eval_result/ycb/EFN6D_iterative_result**: Evaluation result on YCB_Video dataset with iterative refinement.
		* **experiments/eval_result/linemod**: Evaluation results on LineMOD dataset with iterative refinement.
	* **experiments/logs/**: Training log files.
	* **experiments/scripts**
		* **experiments/scripts/train_ycb.sh**: Training script on the YCB_Video dataset.
		* **experiments/scripts/train_linemod.sh**: Training script on the LineMOD dataset.
		* **experiments/scripts/eval_ycb.sh**: Evaluation script on the YCB_Video dataset.
		* **experiments/scripts/eval_linemod.sh**: Evaluation script on the LineMOD dataset.
* **download.sh**: Script for downloading YCB_Video Dataset, preprocessed LineMOD dataset and the trained checkpoints.