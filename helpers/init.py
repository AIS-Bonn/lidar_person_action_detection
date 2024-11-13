### Libraries ###

#%load_ext autoreload
#%autoreload 2
import os
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.random as four
from fiftyone import ViewField as F

import torch
import detectron2
from ouster import client, viz

# import some common libraries
import numpy as np
import random
from datetime import datetime
import skimage.io as io

import sys
sys.path.append("..")
sys.path.append("../MaskDINO")

# import some common detectron2 utilities
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config

import maskdino

from helpers.util import get_timestamp, select_classes, merge_to_superclass, concat_datasets, make_archive
from helpers.fiftyone_detectron2_bridge import clean_instances, detectron_to_fo, get_fiftyone_dicts
from helpers.config import add_config

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def initialization(merge_to_one_class = False, channels = [0,1,2,3], swin=True, set_seed=None, encoding = False):

    if set_seed != None:
        seed = set_seed
        seed_everything(seed)

    # name for the dataset
    name = "person_detection_lidar"
    
    # The directory containing the dataset to import
    root = os.getcwd()
    
    data_dir = os.path.join(root, 'images')
    
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'
    
    difficult_train_dir = data_dir + '/difficult/difficult_train'
    difficult_test_dir = data_dir + '/difficult/difficult_test'
    difficult_valid_dir = data_dir + '/difficult/difficult_valid'
    
    
    labels_path = os.path.join(root, 'annotations')
    
    train_labels_path = labels_path + '/train.json'
    test_labels_path = labels_path + '/test.json'
    valid_labels_path = labels_path + '/valid.json'
    
    difficult_train_labels_path = labels_path + '/difficult_train.json'
    difficult_test_labels_path = labels_path + '/difficult_test.json'
    difficult_valid_labels_path = labels_path + '/difficult_valid.json'
    
    # setup the fiftyone dataset
    dataset_type = fo.types.COCODetectionDataset
    
    try:
        train_dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=train_dir,
            labels_path=train_labels_path,
            name=name+'_train',
        )
    except ValueError: 
        fo.delete_dataset(name+'_train')
        train_dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=train_dir,
            labels_path=train_labels_path,
            name=name+'_train',
        )
        
    try:
        test_dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=test_dir,
            labels_path=test_labels_path,
            name=name+'_test',
        )
    except ValueError: 
        fo.delete_dataset(name+'_test')
        test_dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=test_dir,
            labels_path=test_labels_path,
            name=name+'_test',
        )
    
    try:
        valid_dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=valid_dir,
            labels_path=valid_labels_path,
            name=name+'_valid',
        )
    except ValueError: 
        fo.delete_dataset(name+'_valid')
        valid_dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=valid_dir,
            labels_path=valid_labels_path,
            name=name+'_valid',
        )
    
    print('Found train dataset labels:')
    print(train_dataset.distinct("segmentations.detections.label"))
    print('Found test dataset labels:')
    print(test_dataset.distinct("segmentations.detections.label"))
    print('Found valid dataset labels:')
    print(valid_dataset.distinct("segmentations.detections.label"))
    
    
    try:
        difficult_train_dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=difficult_train_dir,
            labels_path=difficult_train_labels_path,
            name=name+'_difficult_train',
        )
    except ValueError: 
        fo.delete_dataset(name+'_difficult_train')
        difficult_train_dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=difficult_train_dir,
            labels_path=difficult_train_labels_path,
            name=name+'_difficult_train',
        )
    
    try:
        difficult_test_dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=difficult_test_dir,
            labels_path=difficult_test_labels_path,
            name=name+'_difficult_test',
        )
    except ValueError: 
        fo.delete_dataset(name+'_difficult_test')
        difficult_test_dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=difficult_test_dir,
            labels_path=difficult_test_labels_path,
            name=name+'_difficult_test',
        )
    
    try:
        difficult_valid_dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=difficult_valid_dir,
            labels_path=difficult_valid_labels_path,
            name=name+'_difficult_valid',
        )
    except ValueError: 
        fo.delete_dataset(name+'_difficult_valid')
        difficult_valid_dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=difficult_valid_dir,
            labels_path=difficult_valid_labels_path,
            name=name+'_difficult_valid',
        )
        
    print('Found difficult_train dataset labels:')
    print(difficult_train_dataset.distinct("segmentations.detections.label"))
    print('Found difficult_test dataset labels:')
    print(difficult_test_dataset.distinct("segmentations.detections.label"))
    print('Found difficult_valid dataset labels:')
    print(difficult_valid_dataset.distinct("segmentations.detections.label"))
    
    
    bad_images = [173, 174, 510, 553, 576, 577, 578, 583, 584, 589, 596, 606, 609, 613, 624, 625]
    datasets = [train_dataset, test_dataset, valid_dataset, difficult_train_dataset, difficult_test_dataset, difficult_valid_dataset]
    dataset_names = ['train', 'test', 'valid', 'difficult_train', 'difficult_test', 'difficult_valid']
    for i, dataset in enumerate(datasets):
        dataset_name = dataset_names[i]
        for image in bad_images:
            image_path = os.path.join(os.getcwd(), 'images', dataset_name, 'image_' + str(image) + '.png')
            dataset = dataset.match(F("filepath") != image_path)
            
    
    train_dataset_combined = concat_datasets([train_dataset, difficult_train_dataset])
    test_dataset_combined = concat_datasets([test_dataset, difficult_test_dataset])
    valid_dataset_combined = concat_datasets([valid_dataset, difficult_valid_dataset])
    
    # print(train_dataset_combined)
    # print(test_dataset_combined)
    # print(valid_dataset_combined)
    
    
    four_channel_means = []
    four_channel_stds = []
    
    for root, dirs, files in os.walk(data_dir):
        
        for image in files:
            
            values = io.imread(os.path.join(root, image))
            
            means = []
            stds = []
            
            for channel in range(len(channels)):
                
                means.append(np.mean(values[:,:,channel]))
                stds.append(np.std(values[:,:,channel]))

            if encoding:
                metadata_path = os.path.join('include', 'metadata.json')
            
                with open(metadata_path, 'r') as f:
                    metadata = client.SensorInfo(f.read())
    
                xyzlut = client.XYZLut(metadata)
                
                range_values = (255 - values[:,:,3])*(2**8)
                range_values = np.flipud(range_values)
    
                range_values = client.destagger(metadata, range_values, True)
    
                xyz = xyzlut(range_values)
                xyz = np.reshape(xyz, (-1, 3))
                a = np.sqrt(2)/2
                rotation = np.array(((a,0, a), (0, 1, 0), (-a, 0, a)), dtype = 'float64')
                xyz =  (rotation @ xyz.T).T
                xyz[:,2] = xyz[:,2].max() - xyz[:,2] + xyz[:,2].min()
    
                xyz = np.reshape(xyz, (values.shape[0], values.shape[1], 3))

                for channel in range(xyz.shape[2]):
                    
                    means.append(np.mean(xyz[:,:,channel]))
                    stds.append(np.std(xyz[:,:,channel]))
    
                
            four_channel_means.append(means)
            four_channel_stds.append(stds)
        
    four_channel_means = np.array(four_channel_means)
    four_channel_stds = np.array(four_channel_stds)
    
    four_channel_means = np.mean(four_channel_means, axis = 0)
    four_channel_stds = np.mean(four_channel_stds, axis = 0)
    
    # print("Means of the dataset: ", four_channel_means)
    # print("STDs of the dataset: ", four_channel_stds)
    

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_config(cfg)

    cfg.MODEL.CHANNELS = channels

    if swin:
        cfg.merge_from_file('include/swin_config.yaml')
    else:
        cfg.merge_from_file('include/r50_config.yaml')

    cfg.MODEL.PIXEL_MEAN = list(four_channel_means)
    cfg.MODEL.PIXEL_STD = list(four_channel_stds)

    if merge_to_one_class:
        # merging to one label
        old_labels = train_dataset_combined.distinct("segmentations.detections.label")
        new_labels = ['person', 'person', 'person']
        
        train_dataset_combined = merge_to_superclass(train_dataset_combined, new_labels, old_labels)
        valid_dataset_combined = merge_to_superclass(valid_dataset_combined, new_labels, old_labels)
        test_dataset_combined = merge_to_superclass(test_dataset_combined, new_labels, old_labels)
    
    # bridge from fiftyone to detectron2 and sanity-checking the constructed dataset
    
    classes = train_dataset_combined.distinct("segmentations.detections.label")
    labels_dict = {class_label: class_idx for class_idx, class_label in enumerate(classes)}
    print('Label dictionary for complete dataset')
    print(labels_dict)

    if merge_to_one_class:
        for dataset, tag in [(train_dataset_combined, "train"), (valid_dataset_combined, "valid"), (test_dataset_combined, "test")]:
            view = dataset
            if "fiftyone_person_" + tag in DatasetCatalog.list():
                DatasetCatalog.remove("fiftyone_person_" + tag)
            DatasetCatalog.register("fiftyone_person_" + tag, lambda view=view: get_fiftyone_dicts(view, labels_dict))
            MetadataCatalog.get("fiftyone_person_" + tag).set(thing_classes=classes, evaluator_type="coco")
        metadata = MetadataCatalog.get("fiftyone_person_train")

        cfg.DATASETS.TRAIN = ('fiftyone_person_train',)
        cfg.DATASETS.TEST = ('fiftyone_person_valid',)
        
    else:
        for dataset, tag in [(train_dataset_combined, "train"), (valid_dataset_combined, "valid"), (test_dataset_combined, "test")]:
            view = dataset
            if "fiftyone_action_" + tag in DatasetCatalog.list():
                DatasetCatalog.remove("fiftyone_action_" + tag)
            DatasetCatalog.register("fiftyone_action_" + tag, lambda view=view: get_fiftyone_dicts(view, labels_dict))
            MetadataCatalog.get("fiftyone_action_" + tag).set(thing_classes=classes, evaluator_type="coco")
        metadata = MetadataCatalog.get("fiftyone_action_train")
        
        cfg.DATASETS.TRAIN = ('fiftyone_action_train',)
        cfg.DATASETS.TEST = ('fiftyone_action_valid',)
    
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(classes)

    cfg.MODEL.CHANNELS = channels

    if set_seed != None:
        cfg.SEED = seed

    cfg.MODEL.ENCODING = encoding

    #change for your lidar
    cfg.LIDAR_ANGLE = 45
    cfg.FLIP = True

    return cfg, classes, labels_dict, train_dataset_combined, test_dataset_combined, valid_dataset_combined