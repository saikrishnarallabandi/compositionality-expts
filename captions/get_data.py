import os
from pycocotools.coco import COCO


instances_annFile = "cocoapi/annotations/instances_train2014.json"
coco = COCO(instances_annFile)
captions_annFile = 'cocoapi/annotations/captions_train2014.json'
coco_caps = COCO(captions_annFile)
ids = list(coco.anns.keys())
#annIds = coco_caps.getAnnIds(imgIds=249471);
#anns = coco_caps.loadAnns(annIds)
#coco_caps.showAnns(anns)
