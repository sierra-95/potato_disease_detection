
# Potato Disease Detection 

This project contains two main parts:

1. **Model Development (PyTorch)**
   - Dataset collection and preparation from Kaggle.
   - Model training scripts, evaluation and testing.
   - Saved trained model weights.

2. **ROS 2 Humble Package**
   - Real-time image inference pipeline.
   - Topics for image subscription, inference, and result publishing.

## Setting up your environment
```bash
#install dependancies -- might take some time
pip install torch torchvision
#clone the repository
git clone https://github.com/sierra-95/potato_disease_detection.git
cd potato_disease_detection
```

## Dataset Preparation

⚠️ **Warning:** <span style="color:orange">If you do not intend to train your own model, skip to **Model testing**.</span>

1. **Download the Dataset**  
   Get the dataset from [Kaggle – Plant Disease](https://www.kaggle.com/datasets/emmarex/plantdisease).

2. **Create the Directory Structure**  
   ```bash
   cd dataset
   mkdir -p train/Early_Blight train/Healthy train/Late_Blight \
            val/Early_Blight   val/Healthy   val/Late_Blight
3. **Use the provided Python script to split the dataset into training and validation sets:**
    
    Edit this the paths in this file to match yours
    ```bash
    python3 split_dataset.py
    ```


## Model Training

### Version 1 -> [Basic Train](train_basic.py)

The first version of the model [model v1](models/model_v1.pth) used a standard **ResNet18** architecture trained on the Kaggle dataset:

* Pre-trained ResNet18 backbone
* 10 epochs, batch size 32

However, due to limited *Healthy* samples, the model underperformed on healthy leaves.

---

### version 2 -> [Weighted Train](train_weighted.py)

To fix class imbalance, the second version [model v2](models/model_v2.pth) introduced **Weighted Sampling** and **data augmentation**:

* Added random rotation, color jitter, and horizontal flip.
* Used `WeightedRandomSampler` to balance classes.
* Trained for 20 epochs.


## Model Testing 

### Standalone

A [test script](test/test_model.py) allows running inference **without ROS 2** for debugging and validation.
It reads test images from [here](potato_disease_detection/images/early.png) and prints predicted class results on the terminal.

### ROS 2 Integration
```bash
#navigate to repository root then proceed
colcon build --symlink-install
source install/setup.bash
```
#### Publisher
```bash
#If you have a camera - this publishes /camera/image/compressed
ros2 run potato_disease_detection camera_publisher
#----------------------------------------------------------------
#if no camera  - this publishes an image to /potato_image
ros2 run potato_disease_detection image_publisher
```
#### Subscribers

```bash
ros2 launch potato_disease_detection model.launch.py
```

The launch file starts:

* **Camera bridge**
  Subscribes to `/camera/image/compressed` and republishes to `/potato_image`.
* **Camera viewer**
  Displays live camera feed using OpenCV cv2.imshow.
* **Inference engine**
  Subscribes to `/potato_image`, runs inference using `model_v2.pth`, and publishes results to `/inference_result`.



Built with ❤️ by **Sierra-95**