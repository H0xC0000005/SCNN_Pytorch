# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import numpy
import numpy as np
import tensorflow as tf


# import numpy as np
# import scipy.misc
#
# # may throw import error for py2
# from io import BytesIO  # Python 3.x


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        """
        migrate from v1 to v2: use this IO
        tf.summary.create_file_writer(
            logdir,
            max_queue=None,
            flush_millis=None,
            filename_suffix=None,
            name=None,
            experimental_trackable=False
        )
        """
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        """
        deprecated: new tf has tf.summary for scalar, use API
        tf.summary.scalar(
            name, data, step=None, description=None
        )
        
        step: used to summary at timestamp / step for time series
        """
        with self.writer.as_default(step=step):
            tf.summary.scalar(tag, value, step=step)

    def image_summary(self, tag, images, step, greyscale=True):
        """Log a list of images."""
        """
        same, use new API 
        tf.summary.image(
            name, data, step=None, max_outputs=3, description=None
        )
        
        data should be array of image (images here) or images in 4 dim 
        image only receive 4 dim input as [batch channel height width]
        """
        if isinstance(images, list):
            """ cat list of img to 1st dim """
            image_ndarr: np.ndarray
            for idx, image in enumerate(images):
                assert isinstance(image, np.ndarray), f"in image_summary image element is {type(image)}" \
                                                      f"instead of np.ndarray"
                if len(image.shape) == 2 and greyscale:
                    # greyscale image
                    # add extra 1st dim as [1 height width] for stacking
                    images[idx] = np.stack([image], axis=0)
                else:
                    assert len(image.shape) == 3, f"in image_summary image has dimension {image.shape} neq to " \
                                                  f"desired rank 2 or 3"
                """
                numpy.stack(arrays, axis=0, out=None)[source]
                """
            image_ndarr = numpy.stack(images, axis=0)
        elif isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                # a single multichannel image doesn't need further processing
                image_ndarr = numpy.stack([images], axis=0)
                if greyscale:
                    # may be grayscale that has 1st dim as batch, hence need to roll back to [batch 1 height width]
                    image_ndarr = np.moveaxis(image_ndarr, 1, 0)
            else:
                assert len(images.shape) == 4, f"in image_summary image is a nparray but has shape {images.shape} " \
                                               f"without desired dimension 4"
                image_ndarr = images
        else:
            raise TypeError(f"images has type {type(images)} instead of ndarr or list of ndarr")

        """
        torch use [batch channel height width] while tensorboard use [batch height width channel] 
        hence need to convert by np.moveaxis
        
        numpy.moveaxis(a, source, destination)[source]
        """

        if image_ndarr.shape[1] in (1, 2, 3, 4):
            # sec dim is channel, roll to the last dim
            image_ndarr = np.moveaxis(image_ndarr, 1, -1)
        else:
            raise ValueError(f"in image_summary, image_ndarr has dim {image_ndarr.shape} that has sec dimension "
                             f"not its channel ")

        print(f">>>>>>>>>>>>>>>>>> image_ndarr has shape {image_ndarr.shape}")
        with self.writer.as_default(step=step):
            tf.summary.image(tag, image_ndarr, step=step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        """
        same, use this API
        
        tf.summary.histogram(
            name, data, step=None, buckets=None, description=None
        )
        """
        with self.writer.as_default(step=step):
            tf.summary.histogram(tag, values, step=step, buckets=bins)
