#vizualization of the the point cloud from range image with ground truth mask

from ouster import client, viz
import os
import skimage.io as io
import numpy as np
import math
from pycocotools.coco import COCO
import json

metadata_path = os.path.join('include', 'metadata.json')
image_path = os.path.join('images', 'train', 'image_112.png')
annotations_path = os.path.join('annotations', 'train.json')
image_id = 113
annotations_new_path = os.path.join('output', 'test', 'inference', 'train_labels.json')
image_id_new = 11

coco = COCO(annotations_path)
with open(annotations_path) as f:
    data = json.load(f)
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(catIds=cat_ids, imgIds=image_id)
    anns = coco.loadAnns(anns_ids)

mask = coco.annToMask(anns[0])

coco_new  = COCO(annotations_new_path)
with open(annotations_new_path) as f:
    data_new = json.load(f)
    cat_ids_new = coco_new.getCatIds()
    anns_ids_new = coco_new.getAnnIds(catIds=cat_ids_new, imgIds=image_id_new)
    anns_new = coco_new.loadAnns(anns_ids_new)

mask_new = coco_new.annToMask(anns_new[0])

with open(metadata_path, 'r') as f:
    metadata = client.SensorInfo(f.read())

xyzlut = client.XYZLut(metadata)

image = io.imread(image_path)
image = np.flipud(image)
range = (255 - image[:,:,3])*(2**8)
range = client.destagger(metadata, range, True)

mask = np.flipud(mask)
mask = client.destagger(metadata, mask, True)

mask_new = np.flipud(mask_new)
mask_new = client.destagger(metadata, mask_new, True)
mask_new = mask_new*2
mask = (mask+mask_new)/3

xyz = xyzlut(range)
xyz = np.reshape(xyz, (-1, 3))
a = 45
a = a/360*2*math.pi
rotation = np.array(((np.cos(a),0, np.sin(a)), (0, 1, 0), (-np.sin(a), 0, np.cos(a))), dtype = 'float64')
xyz =  (rotation @ xyz.T).T
xyz[:,2] = xyz[:,2].max() - xyz[:,2] + xyz[:,2].min() 

point_viz = viz.PointViz("Example Viz")
viz.add_default_controls(point_viz)

x_ = np.array([1, 0, 0]).reshape((-1, 1))
y_ = np.array([0, 1, 0]).reshape((-1, 1))
z_ = np.array([0, 0, 1]).reshape((-1, 1))

axis_n = 100
line = np.linspace(0, 1, axis_n).reshape((1, -1))

# basis vector to point cloud
axis_points = np.hstack((x_ @ line, y_ @ line, z_ @ line)).transpose()

# colors for basis vectors
axis_color_mask = np.vstack((np.full(
    (axis_n, 4), [1, 0.1, 0.1, 1]), np.full((axis_n, 4), [0.1, 1, 0.1, 1]),
                             np.full((axis_n, 4), [0.1, 0.1, 1, 1])))

cloud_axis = viz.Cloud(axis_points.shape[0])
cloud_axis.set_xyz(axis_points)
cloud_axis.set_key(np.full(axis_points.shape[0], 0.5))
cloud_axis.set_mask(axis_color_mask)
cloud_axis.set_point_size(3)
point_viz.add(cloud_axis)

cloud_xyz = viz.Cloud(xyz.shape[0])
cloud_xyz.set_xyz(xyz)
cloud_xyz.set_key(mask.ravel())
point_viz.add(cloud_xyz)

point_viz.update()

point_viz.run()