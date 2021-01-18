from enum import Enum

class ModelState(Enum):

    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'

class TrainingState(Enum):

    TRAIN_START = 'train_start'
    TRAIN_END = 'train_end'
    
    EPOCH_START = 'epoch_start'
    EPOCH_END = 'epoch_end'
    
    TRAIN_EPOCH_START = 'train_epoch_start'
    TRAIN_EPOCH_END = 'train_epoch_end'
    
    VALID_EPOCH_START = 'valid_epoch_start'
    VALID_EPOCH_END = 'valid_epoch_end'
    
    TRAIN_STEP_START = 'train_step_start'
    TRAIN_STEP_END = 'train_step_end'
    
    VALID_STEP_START = 'valid_step_start'
    VALID_STEP_END = 'valid_step_end'
    
    TEST_STEP_START = 'on_test_step_start'
    TEST_STEP_END = 'on_test_step_end'