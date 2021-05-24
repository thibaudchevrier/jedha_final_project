import tensorflow as tf
import numpy as np
from mask_rcnn_matterport.utils import norm_boxes, denorm_boxes, unmold_mask, resize_image, generate_pyramid_anchors
from mask_rcnn_matterport.visualize import random_colors
import json
import skimage
import math
import csv

class Model():

    def __init__(self):
        with open("mask_rcnn_matterport/deployement/config.json") as json_file:
            self.data = json.load(json_file)
        self.model = tf.keras.models.load_model("mask_rcnn_matterport/deployement/", compile=False)
        self.range_color = random_colors(self.data.get("NUM_CLASSES") + 1)
    
    def compute_backbone_shapes(self, image_shape):
        """Computes the width and height of each stage of the backbone network.

        Returns:
            [N, (height, width)]. Where N is the number of stages
        """
        if callable(self.data.get("BACKBONE")):
            return self.data.get("COMPUTE_BACKBONE_SHAPE")(image_shape)

        # Currently supports ResNet only
        assert self.data.get("BACKBONE") in ["resnet50", "resnet101"]
        return np.array(
            [[int(math.ceil(image_shape[0] / stride)),
                int(math.ceil(image_shape[1] / stride))]
                for stride in self.data.get("BACKBONE_STRIDES")])
    
    @staticmethod
    def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
        """Takes attributes of an image and puts them in one 1D array.

        image_id: An int ID of the image. Useful for debugging.
        original_image_shape: [H, W, C] before resizing or padding.
        image_shape: [H, W, C] after resizing and padding
        window: (y1, x1, y2, x2) in pixels. The area of the image where the real
                image is (excluding the padding)
        scale: The scaling factor applied to the original image (float32)
        active_class_ids: List of class_ids available in the dataset from which
            the image came. Useful if training on images from multiple datasets
            where not all classes are present in all datasets.
        """
        meta = np.array(
            [image_id] +                  # size=1
            list(original_image_shape) +  # size=3
            list(image_shape) +           # size=3
            list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
            [scale] +                     # size=1
            list(active_class_ids)        # size=num_classes
        )
        return meta
    
    def mold_image(self, images):
        """Expects an RGB image (or array of images) and subtracts
        the mean pixel and converts it to float. Expects image
        colors in RGB order.
        """
        return images.astype(np.float32) - self.data.get("MEAN_PIXEL")

    @staticmethod
    def unmold_detections(detections, mrcnn_mask, original_image_shape,
                        image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            molded_image, window, scale, padding, crop = resize_image(
                image,
                min_dim=self.data.get("IMAGE_MIN_DIM"),
                min_scale=self.data.get("IMAGE_MIN_SCALE"),
                max_dim=self.data.get("IMAGE_MAX_DIM"),
                mode=self.data.get("IMAGE_RESIZE_MODE")
            )
            molded_image = self.mold_image(molded_image)
            # Build image_meta
            image_meta = self.compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.data.get("NUM_CLASSES")], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows


    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = self.compute_backbone_shapes(image_shape)
        # Cache anchors and reuse if image shape is the same
        _anchor_cache = {}
        if not tuple(image_shape) in _anchor_cache:
            # Generate Anchors
            a = generate_pyramid_anchors(
                self.data.get("RPN_ANCHOR_SCALES"), 
                self.data.get("RPN_ANCHOR_RATIOS"), 
                backbone_shapes,
                self.data.get("BACKBONE_STRIDES"), 
                self.data.get("RPN_ANCHOR_STRIDE")
            )
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # Normalize coordinates
            _anchor_cache[tuple(image_shape)] = norm_boxes(a, image_shape[:2])
        return _anchor_cache[tuple(image_shape)] 
        
    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        images_input = [skimage.io.imread(images)]
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images_input)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.data.get("BATCH_SIZE"),) + anchors.shape)

        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images_input):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                        image.shape, molded_images[i].shape,
                                        windows[i])
            results.append({
                "image_id": images,
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results