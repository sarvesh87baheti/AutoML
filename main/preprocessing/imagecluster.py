import cv2
import numpy as np
from sklearn.cluster import KMeans
import argparse

def kmeans_image_segmentation(input_path, output_path, n_clusters):
    # Load image
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape image to 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)

    # Replace each pixel with centroid color
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_pixels.reshape(image.shape).astype(np.uint8)

    # Convert back to BGR for saving
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    # Save output image
    cv2.imwrite(output_path, segmented_image)

    print(f"Segmented image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMeans Image Clustering")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("output_image", help="Path to save output image")
    parser.add_argument("n_clusters", type=int, help="Number of color clusters")

    args = parser.parse_args()

    kmeans_image_segmentation(
        args.input_image,
        args.output_image,
        args.n_clusters
    )
