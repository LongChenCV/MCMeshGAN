import re
EPOCHS = 100
IMAGE_LENGTH = 128
IMAGE_DEPTH = 3


BATCH_SIZE = 64
KERNEL_SIZE = 2
STRIDE_SIZE = 2

NUM_ENCODER_CHANNELS = 64
NUM_Z_CHANNELS = 100
NUM_GEN_CHANNELS = 1024

NUM_VERTICES = 10000

NUM_AGES = 100
NUM_GENDERS = 2
NUM_INTERVELS = 100
LABEL_LEN = NUM_AGES + NUM_GENDERS
NUM_GENDERS_EXPANDED = NUM_GENDERS * (NUM_AGES // NUM_GENDERS)
LABEL_LEN_EXPANDED = NUM_AGES + NUM_GENDERS_EXPANDED
NUM_AGES_GENDERS_INTERVELS = NUM_AGES*3

LENGHTH_INPUT = 10000
# LENGHTH_INPUT =2252
MALE = 0
FEMALE = 1

UTKFACE_DEFAULT_PATH = './data/AgeAorta'
UTKFACE_ORIGINAL_IMAGE_FORMAT = re.compile('^(\d+)_(\d+)_\d+_(\d+)\.jpg\.chip\.jpg$')

TRAINED_MODEL_EXT = '.dat'
TRAINED_MODEL_FORMAT = "{}" + TRAINED_MODEL_EXT