from __future__ import print_function
import cv2
import os
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2, i):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Saving top matches in /Matches
    path = 'Matches/'
    out = "match" + str(i)
    out = out + ".jpg"
    #print("Saving Match image : ", out)
    cv2.imwrite(os.path.join(path, out), imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


if __name__ == '__main__':
    # Read Reference image from the folder Original/
    i = 1
    refFilename = "Original/original.jpg"
    #print("Reading reference image "" : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read images from the folder images/
    rootdir = "images/"
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            frame = cv2.imread(os.path.join(subdir, file))
            print("Aligning image "+str(i))
            imReg, h = alignImages(frame, imReference,i)
            path = 'Output/'
            outFilename = "output"+str(i)
            outFilename = outFilename+".jpg"
            #print("Saving aligned image : ", outFilename);
            cv2.imwrite(os.path.join(path, outFilename), imReg)
            i = i + 1

