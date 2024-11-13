from helpers.model_trainer import Predictor
import cv2
from helpers.fiftyone_detectron2_bridge import clean_instances, detectron_to_fo, get_fiftyone_dicts
import os
import fiftyone as fo

def benchmark(cfg, classes, labels_dict, train_dataset_combined, test_dataset_combined, valid_dataset_combined, save=False):
    
    predictor = Predictor(cfg)
    
    # bridge back from detectron2 to fiftyone to annotate the validation data with model predictions
    
    # this is just a quick summary of model performance as a final sanity-check before deployment
    # in addition, the training process above yields validation map scores as a side result
    
    print("TEST DATASET\n")
    
    test_view = test_dataset_combined
    dataset_dicts = get_fiftyone_dicts(test_view, labels_dict)
    predictions = {}
    for d in dataset_dicts:
        img_w = d["width"]
        img_h = d["height"]
        img = cv2.imread(d["file_name"], cv2.IMREAD_UNCHANGED)
        outputs = predictor(img)
        detections, instances = detectron_to_fo(outputs, img_w, img_h, classes, score_tresh=0.3)
        predictions[d["image_id"]] = detections
    
    test_dataset_combined.set_values("predictions", predictions, key_field="id")

    if save:
        export_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
        dataset_type = fo.types.COCODetectionDataset  
        test_dataset_combined.export(
            export_dir=export_dir,
            dataset_type=dataset_type,
            label_field="predictions",
            export_media=False,
            labels_path="test_labels.json"
        )
    
    test_results = test_dataset_combined.evaluate_detections(
        "predictions",
        gt_field="segmentations",
        eval_key="eval",
        use_masks=True,
        compute_mAP=True,
        classes=test_dataset_combined.distinct("segmentations.detections.label")
    )
    test_results.print_report()
    
    
    print("VALIDATION DATASET\n")
    
    val_view = valid_dataset_combined
    dataset_dicts = get_fiftyone_dicts(val_view, labels_dict)
    predictions = {}
    for d in dataset_dicts:
        img_w = d["width"]
        img_h = d["height"]
        img = cv2.imread(d["file_name"], cv2.IMREAD_UNCHANGED)
        outputs = predictor(img)
        detections, instances = detectron_to_fo(outputs, img_w, img_h, classes, score_tresh=0.3)
        predictions[d["image_id"]] = detections
    
    valid_dataset_combined.set_values("predictions", predictions, key_field="id")

    if save:
        export_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
        dataset_type = fo.types.COCODetectionDataset  
        valid_dataset_combined.export(
            export_dir=export_dir,
            dataset_type=dataset_type,
            label_field="predictions",
            export_media=False,
            labels_path="valid_labels.json"
        )
    
    val_results = valid_dataset_combined.evaluate_detections(
        "predictions",
        gt_field="segmentations",
        eval_key="eval",
        use_masks=True,
        compute_mAP=True,
        classes=valid_dataset_combined.distinct("segmentations.detections.label")
    )
    val_results.print_report()
    
    
    print("TRAIN DATASET\n")
    
    train_view = train_dataset_combined
    dataset_dicts = get_fiftyone_dicts(train_view, labels_dict)
    predictions = {}
    for d in dataset_dicts:
        img_w = d["width"]
        img_h = d["height"]
        img = cv2.imread(d["file_name"], cv2.IMREAD_UNCHANGED)
        outputs = predictor(img)
        detections, instances = detectron_to_fo(outputs, img_w, img_h, classes, score_tresh=0.3)
        predictions[d["image_id"]] = detections
    
    train_dataset_combined.set_values("predictions", predictions, key_field="id")

    if save:
        export_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
        dataset_type = fo.types.COCODetectionDataset  
        train_dataset_combined.export(
            export_dir=export_dir,
            dataset_type=dataset_type,
            label_field="predictions",
            export_media=False,
            labels_path="train_labels.json"
        )
    
    train_results = train_dataset_combined.evaluate_detections(
        "predictions",
        gt_field="segmentations",
        eval_key="eval",
        use_masks=True,
        compute_mAP=True,
        classes=train_dataset_combined.distinct("segmentations.detections.label")
    )
    train_results.print_report()