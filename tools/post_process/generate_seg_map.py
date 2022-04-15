import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
import os

def cocoSegmentationToSegmentationMap(coco, imgId, checkUniquePixelLabel=True, includeCrowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the id of the COCO image
    :param checkUniquePixelLabel: (optional) whether every pixel can have at most one label
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: labelMap - [h x w] segmentation map that indicates the label of each pixel
    '''

    # Init
    curImg = coco.imgs[imgId]
    imageSize = (curImg['height'], curImg['width'])
    labelMap = np.zeros(imageSize)

    # Get annotations of the current image (may be empty)
    imgAnnots = [a for a in coco.anns.values() if a['image_id'] == imgId]
    if includeCrowd:
        annIds = coco.getAnnIds(imgIds=imgId)
    else:
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
    imgAnnots = coco.loadAnns(annIds)

    # Combine all annotations of this image in labelMap
    #labelMasks = mask.decode([a['segmentation'] for a in imgAnnots])
    for a in range(0, len(imgAnnots)):
        labelMask = coco.annToMask(imgAnnots[a]) == 1
        #labelMask = labelMasks[:, :, a] == 1
        newLabel = imgAnnots[a]['category_id']
        if checkUniquePixelLabel and (labelMap[labelMask] != 0).any():
            raise Exception('Error: Some pixels have more than one label (image %d)!' % (imgId))

        labelMap[labelMask] = newLabel

    return labelMap

def cocoSegmentationToPng(coco, imgId, pngPath, includeCrowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map and write it to disk.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the COCO id of the image (last part of the file name)
    :param pngPath: the path of the .png file
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: None
    '''

    # Create label map
    labelMap = cocoSegmentationToSegmentationMap(coco, imgId, checkUniquePixelLabel=False, includeCrowd=includeCrowd)
    labelMap = labelMap.astype(np.uint8)

    cv2.imwrite(pngPath, labelMap)

if __name__ == '__main__':
    pngFolder = '/data/raw_data/train_dataset/200/seg_map/images'
    anno_file = '/data/raw_data/train_dataset/200/data.json'
    # Create output folder
    if not os.path.exists(pngFolder):
        os.makedirs(pngFolder)

    # Initialize COCO ground-truth API
    coco = COCO(anno_file)
    imgIds = coco.getImgIds()

    # Convert each image to a png
    imgCount = len(imgIds)
    for i in tqdm(range(0, imgCount)):
        imgId = imgIds[i]
        imgName = coco.loadImgs(ids=imgId)[0]['file_name'].replace('.jpg', '').replace('.png', '')
        # print('Exporting image %d of %d: %s' % (i + 1, imgCount, imgName))
        segmentationPath = '%s/%s.png' % (pngFolder, os.path.basename(imgName))
        cocoSegmentationToPng(coco, imgId, segmentationPath)
