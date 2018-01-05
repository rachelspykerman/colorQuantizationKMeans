import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

# Use k-Means Clustering (MiniBatchKMeans) to perform color quantization
# Using MiniBatchKMeans since this is faster than normal K Means Clustering. MiniBatchKMeans operates on small
# “batches” of the dataset, whereas K-Means operates on the population of the dataset, thus making the mean
# calculation of each centroid, as well as the centroid update loop, much slower

# K Means Clustering partitions n data points into k clusters. Each of the n data points will be assigned to a cluster
# with the nearest mean. The mean of each cluster is called its “centroid” or “center”.
# Overall, applying k-means yields k separate clusters of the original n data points. Data points inside a
# particular cluster are considered to be “more similar” to each other than data points that belong to other clusters.

# number of colors to quantize by
# usually only takes 16 colors to make a good representation of the original image
clusterNum = 10
camera = cv2.VideoCapture(0)

while True:
    ret,frame = camera.read()

    if ret:
        (h,w) = frame.shape[:2]

        # convert frame to L*A*B* color space -- we are clustering using k-means which is based on the euclidean
        # distance, L*a*b* color space is used where the euclidean distance implies perceptual meaning
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)

        # reshape the image to a 2D array instead of 3D --> from M,N,3 to M*N,3
        image = np.reshape(frame, (frame.shape[0] * frame.shape[1], 3))

        # apply k-means using the specified number of clusters and then create the quantized image based on predictions
        # predict what quantized color each pixel in the original image is going to be. This prediction is handled
        # by determining which centroid the input pixel is closest to.
        clt = MiniBatchKMeans(n_clusters=clusterNum)
        labels = clt.fit_predict(image)

        # K Means Way
        #clt = KMeans(n_clusters=clusterNum)
        # fit() method clusters the list of pixels
        #labels = clt.fit_predict(image)

        quantize = clt.cluster_centers_.astype("uint8")[labels]

        # reshape the feature vectors back to images
        quantize = quantize.reshape((h, w, 3))
        image = image.reshape((h, w, 3))

        # convert from L*a*b* to RGB
        quantize = cv2.cvtColor(quantize, cv2.COLOR_LAB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

        # display the images and wait for a keypress
        cv2.imshow("image", np.hstack([frame, quantize]))

        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
# release the video capture
camera.release()
cv2.destroyAllWindows()


# The fewer number of clusters, the less colors used to represent the image so more color loss. However, the script
# runs significantly faster.
#
# As the number of clusters increases, so does the amount of time it takes to perform the clustering. Secondly,
# as the number of clusters increases, so does the amount of memory it takes to store the output image; however,
# in both cases, the memory footprint will still be smaller than the original image since you are working with a
# substantially smaller color palette.
# Downside to this method is that we need to pick a cluster number ahead of time

# Two ways to determine colors in a picture:
    # 1) Can use histograms to determine colors
    # 2) Can use K Means to determine color clusters