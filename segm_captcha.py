#!/usr/bin/env python
###########################################
# Segment 4-chars captchas into four pieces
#
# Author: zhaoyafei0210@gmail.com
###########################################


import numpy as np

import cv2
# import tesseract

# import matplotlib
# from matplotlib import pyplot as plt


def read_and_segment(img_path, to_gray=False):
    print "---> Process image: ", img_path
    img = cv2.imread(img_path, 0)
    cv2.imshow('src', img)
    # cv2.waitKey(0)

    print 'img.shape: ', img.shape

    seg_regions = segment_captcha(img)

    return seg_regions


def segment_captcha(img):
    if img.shape[-1] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    ht, wd = gray.shape

    # plt.close('all')

    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)

    ret, threshed_img = cv2.threshold(gray, 231, 255, cv2.THRESH_BINARY_INV)
    # print 'threshed_img.shape: ', threshed_img.shape
    # cv2.imshow('threshold', threshed_img)

    # sum_rows = np.sum(threshed_img, 1)
    sum_cols = np.sum(threshed_img, 0)

    # print 'sum_rows.shape: ', sum_rows.shape
    # print 'sum_cols.shape: ', sum_cols.shape

    # plt.figure()
    # plt.plot(sum_rows, np.arange(0, ht))
    # plt.show()

    plt.figure()
    plt.plot(np.arange(0, wd), sum_cols)
    plt.show(block=False)

    horiz_zxp = find_zero_crossings(sum_cols)
    print 'horizontal zero crossings: ', horiz_zxp

    horiz_segs = find_seg_points(sum_cols)
    print 'horizontal segm points: ', horiz_segs

    sub_regions = []
    crop_line = np.ones((ht, 10), dtype=threshed_img.dtype) * 128

    seg_regions = []

    for i in range(0, 4):
        sub_region = threshed_img[:, horiz_segs[i * 2]:horiz_segs[i * 2 + 1]]
        sub_regions.append(sub_region)
        sub_regions.append(crop_line)

        sum_rows_sub = np.sum(sub_region, 1)
        vert_zxp = find_zero_crossings(sum_rows_sub)
        print 'vertical zero crossings for sub regions: ', vert_zxp

        seg_y1 = 0
        seg_y2 = ht - 1

        if vert_zxp.size > 1:
            seg_y1 = vert_zxp[0]
            seg_y2 = vert_zxp[-1]
        elif vert_zxp.size == 1:
            if sum_rows_sub[0] > 0:
                seg_y2 = vert_zxp[0]
            else:
                seg_y1 = vert_zxp[0]

        seg_region = sub_region[seg_y1:seg_y2 + 1, :]
        seg_regions.append(seg_region)
        print '---> segm shape', seg_region.shape
        # cv2.imshow('segm' + str(i), seg_region)

    # cv2.imshow('sub_regions', np.hstack(sub_regions))

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return seg_regions


def resize_regions(regions, wd=40, ht=40, to_gray=False):

    resized_regions = []
    # crop_line = np.ones((ht, 10), dtype=threshed_img.dtype) * 128

    for (i, region) in enumerate(regions):
        resized_region = cv2.resize(region, (wd, ht))
        if to_gray:
            resized_region = cv2.cvtColor(resized_region, cv2.COLOR_BGR2GRAY)

        resized_regions.append(resized_region)
        cv2.imshow('resized' + str(i), resized_region)

    # cv2.imshow('resized_regions', np.hstack(resized_regions))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return resized_regions

    # char_list = "abcdefghijklmnopqrstuvwxyz"
    # char_list += char_list.upper()
    # char_list += "0123456789"

    # api = tesseract.TessBaseAPI()
    # api.Init(".", "eng", tesseract.OEM_DEFAULT)
    # api.SetVariable("tessedit_char_whitelist", char_list)
    # api.SetPageSegMode(tesseract.PSM_SINGLE_WORD)
    # tesseract.SetCvImage(threshed_img, api)
    # print api.GetUTF8Text()


def find_zero_crossings(x):
    # print 'x: ', x
    x_max = x.max()
    # print 'x_max: ', x_max
    y = x > x_max * 0.01
    # print 'y: ', y
    zero_crossings = np.where(np.diff(y))[0]
    return zero_crossings


def find_seg_points(x):
    zxp = find_zero_crossings(x)
    print 'zero-cross points: ', zxp
    print 'num of zero-cross points: ', zxp.size

    len = x.size
    eff_len = zxp[-1] - zxp[0]
    est_seg_len = eff_len * 0.25

    if zxp.size == 8:
        return zxp
    elif zxp.size == 6:
        # print zxp[1::2]
        # print zxp[0::2]
        peak_wds = zxp[1::2] - zxp[0::2]
        print 'peak widths: ', peak_wds

        max_peak_idx = np.argmax(peak_wds)
        print 'max peak idx: ', max_peak_idx

        seg_x1, seg_x3 = zxp[max_peak_idx * 2], zxp[max_peak_idx * 2 + 1]

        seg_x2 = find_ostu_threshold(x, seg_x1 + 1, seg_x3 - 1)

        new_seg_pts = np.array([seg_x2, seg_x2])

        new_zxp = np.hstack(
            (zxp[0:max_peak_idx * 2 + 1], new_seg_pts, zxp[max_peak_idx * 2 + 1]))

        return new_zxp
    elif zxp.size == 4:
        # print zxp[1::2]
        # print zxp[0::2]
        peak_wds = zxp[1::2] - zxp[0::2]
        print 'peak widths: ', peak_wds

        max_peak_idx = np.argmax(peak_wds)
        print 'max peak idx: ', max_peak_idx

        seg_x1, seg_x4 = zxp[max_peak_idx * 2], zxp[max_peak_idx * 2 + 1]

        len14 = seg_x4 - seg_x1
        est_seg_len3 = len14 * 0.33

        seg_xn_left = seg_x1
        seg_xn_right = min(seg_x1 + int(est_seg_len3 * 1.75), len - 1)
        seg_x2 = find_ostu_threshold(x, seg_xn_left + 1, seg_xn_right - 1)

        seg_xn_left = max(seg_x4 - int(est_seg_len3 * 1.75), 0)
        seg_xn_right = seg_x4
        seg_x3 = find_ostu_threshold(x, seg_xn_left + 1, seg_xn_right - 1)

        new_seg_pts = np.array(
            [seg_x2, seg_x2, seg_x3, seg_x3])

        new_zxp = np.hstack(
            (zxp[0:max_peak_idx * 2 + 1], new_seg_pts, zxp[max_peak_idx * 2 + 1:]))

        return new_zxp
    else:
        seg_x1, seg_x5 = zxp[0], zxp[1]

        len15 = seg_x5 - seg_x1
        est_seg_len4 = len15 * 0.25

        seg_xn_left = seg_x1
        seg_xn_right = min(seg_x1 + int(est_seg_len4 * 1.75), len - 1)
        seg_x2 = find_ostu_threshold(x, seg_xn_left + 1, seg_xn_right - 1)

        seg_xn_left = max(seg_x5 - int(est_seg_len4 * 1.75), 0)
        seg_xn_right = seg_x5
        seg_x4 = find_ostu_threshold(x, seg_xn_left + 1, seg_xn_right - 1)

        seg_xn_left = max(seg_x2 + int(est_seg_len4 * 0.25), len - 1)
        seg_xn_right = min(seg_x4 - int(est_seg_len4 * 0.25), 0)
        seg_x3 = find_ostu_threshold(x, seg_xn_left + 1, seg_xn_right - 1)

        new_seg_pts = np.array(
            [seg_x2, seg_x2, seg_x3, seg_x3])

        new_zxp = np.hstack(
            (zxp[0:max_peak_idx * 2 + 1], new_seg_pts, zxp[max_peak_idx * 2 + 1:]))

        return new_zxp


def find_ostu_threshold(hist, min_bin, max_bin):
    hist = hist[min_bin:max_bin].astype(float)
    bin_centers = np.arange(min_bin, max_bin)

    # print '---> hist: ', hist

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    # print 'weight1: ', weight1

    weight2 = np.cumsum(hist[::-1])[::-1]

    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    # print 'variance12: ', variance12
    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]

    print 'Otsu threshold is: ', threshold

    return threshold


if __name__ == '__main__':
    # img_path = r'./webcode_01.png'
    # read_and_segment(img_path)

    img_list = r'./list_img.txt'
    with open(img_list, 'r') as fp:
        for line in fp:
            segm_regions = read_and_segment(line.strip())
            resized_regions = resize_regions(segm_regions)
    fp.close()
