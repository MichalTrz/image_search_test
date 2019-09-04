# Learning rate parameters
BASE_LR = 0.001
EPOCH_DECAY = 30 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.1 # factor by which the learning rate is reduced.


# DATASET INFO
NUM_CLASSES = 11 # set the number of classes in your dataset
MARGIN = 0.5
DATA_DIR = './' # path to dataset


# DATALOADER PROPERTIES
BATCH_SIZE = 12 # Set as high as possible. If you keep it too high, you'll get an out of memory error.


### GPU SETTINGS
CUDA_DEVICE = 1 # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 0 # set to 1 if want to run on gpu.


# SETTINGS FOR DISPLAYING ON TENSORBOARD
USE_TENSORBOARD = 0 #if you want to use tensorboard set this to 1.
#TENSORBOARD_SERVER = "http://desktop-a2gi0vc:8889/" # If you set.
TENSORBOARD_SERVER = "http://localhost:9119/"
EXP_NAME = "fine_tuning_experiment" # if using tensorboard, enter name of experiment you want it to be displayed as.
