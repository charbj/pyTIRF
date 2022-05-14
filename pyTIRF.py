'''
###########################################################################################################
            TIRF analysis ***VIEW*** - python programs for analysis of single molecule TIRF
            Developed by Charles Bayly-Jones (2022) - Monash University, Melbourne, Australia
###########################################################################################################
'''
import argparse, os
import textwrap
from skimage import io

from scipy import optimize, signal

import numpy as np
np.set_printoptions(precision=2)
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from algorithms import *
from skimage.restoration import denoise_tv_chambolle

import warnings
warnings.filterwarnings("ignore")

class core():
    def __init__(self, args):
        self.args = args
        self.liposome_decay = 2.25e-3
        self.protein_decay = 1.87e-2
        self.dt = 5

    def gain_ref_per_frame(self, stack, gain_ref):
        cache = []
        for i in range(stack.shape[0]):
            arr = stack[i,:,:] - gain_ref
            cache.append(arr)
        return np.array(cache, dtype=np.float32)

    def read(self, image, start=0, stop=0, full=False):
        if not full:
            cache = []
            for i in range(start, stop):
                arr = io.imread(image, img_num=i)
                cache.append(arr)
            return np.array(cache, dtype=np.float32)
        else:
            return np.array(io.imread(image), dtype=np.float32)

    def write(self, image, name):
        io.imsave(name, image)

    def average(self, image, gain_ref=None):
        #Note to self, changed np.average to np.mean and added float32 - might break code
        if gain_ref is not None:
            return np.mean(image, axis=0, dtype=np.float32) - gain_ref
        else:
            return np.mean(image, axis=0, dtype=np.float32)

    def piecewise_constant(self, trace, boundaries):
        x = np.arange(0,len(trace),1)
        def chunker(seq, step):
            return (seq[pos:pos+step] for pos in range(0, len(seq), step))

        def _multi_step(x, *params):
            sum = np.zeros(x.shape[0])
            for step in chunker(params, 3):
                func = np.zeros(x.shape[0])
                const, start, stop = step
                func[int(start):int(stop)] = const
                sum = np.add(sum, func)
            return sum

        # Fit multiple gaussians to histogram
        def _find_centre(boundary_mask):
            ind = []
            centre = []
            previous = 0
            for ele in range(boundary_mask.shape[0]):
                mask = boundary_mask[ele]
                if mask == 1:
                    ind.append(ele)
                elif mask == 0 and previous == 1:
                    if len(ind) > 1:
                        mid_ind = int(math.ceil(len(ind)/2))
                        centre.append(ind[mid_ind])
                        ind = []
                else:
                    pass
                previous = mask
            return centre

        centres = _find_centre(boundaries)
        if centres:
            p0 = []
            start = 0
            for event in centres:
                # average = np.sum(trace[start:event])/(event-start)
                p0.append([0, start, event])
                start = event
            p0.append([0, start, len(boundaries)])
            p0 = [param for sublist in p0 for param in sublist]
        else:
            p0 = [0, 0, len(boundaries)]

        low_b0 = []
        high_b0 = []
        for param in chunker(p0, 3):
            _, lim_1, lim_2 = param
            low_b0.append([-100, lim_1-1, lim_2-1])
            high_b0.append([100, lim_1+1, lim_2+1])

        low_b0 = [lim for sublist in low_b0 for lim in sublist]
        high_b0 = [lim for sublist in high_b0 for lim in sublist]

        try:
            params_piecewise, pcov_piecewise = optimize.curve_fit(_multi_step,
                                                                  x,
                                                                  trace,
                                                                  p0=p0,
                                                                  bounds=(low_b0, high_b0))
        except RuntimeError:
            print("Failed to fit piecewise constant function.")
            return x
        except ValueError:
            print("Failed due to NaN or inf")
            return x
        else:
            return _multi_step(x, *params_piecewise)

    def find_boundaries(self, trace, window=25, threshold=2):
        boundary = []
        for step in range(window):
            boundary.append(0)

        for step,_ in enumerate(trace):
            step += window
            if step == len(trace) - window:
                break
            past = trace[step-window:step]
            future = trace[step+1:step+window+1]
            sigma_past = np.std(past)
            past_mean = np.sum(past) / window
            future_mean = np.sum(future) / window

            if future_mean >= threshold*sigma_past+past_mean:
                boundary.append(1)
            elif future_mean <= past_mean-threshold*sigma_past:
                boundary.append(1)
            else:
                boundary.append(0)

        for step in range(window):
            boundary.append(0)

        return np.array(boundary)

    def extract_intensities(self, trace, step_function):
        if len(trace) != len(step_function):
            raise ValueError("Trace and step function do not have same length")

        events = []
        c = 1
        sum = trace[0]
        previous = step_function[0]
        for frame,intensity in enumerate(trace):
            sum += intensity
            c += 1
            if step_function[frame] == 0 and previous == 1:
                #print(f"Appending {sum/c} since stepfunction was {step_function[frame]}")
                events.append(sum/c)
                c = 1
                sum = intensity
            elif frame == len(trace)-1:
                events.append(sum/c)

            previous = step_function[frame]

        if events:
            fit = []
            step = 0
            previous = step_function[0]
            for frame,val in enumerate(step_function):
                if step != len(events)-1 and previous != val and previous != 0:
                    step += 1
                fit.append(events[step])
                previous = val

            jumps = []
            for i,_ in enumerate(events[:-1]):
                jump = np.abs(events[i+1] - events[i])
                jumps.append(jump)
        else:
            jumps = []
            fit = []

        return events, jumps, np.array(fit)

    def align(self, ref, image):
        sz = image.shape
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        number_of_iterations = 1000
        termination_eps = 1e-10
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        (cc, warp_matrix) = cv2.findTransformECC(ref, image, warp_matrix, warp_mode, criteria)
        aligned = cv2.warpAffine(image, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned, warp_matrix

    def align_and_register(self, image, image2, ref, ref2, two_ch=False):
        print("Perforing Motion Correction - deblur")
        frames_ch0 = []
        frames_ch1 = []
        warp_ch0 = []
        warp_ch1 = []
        size_ch0 = image.shape[0]
        size_ch1 = image2.shape[0]
        for i in range(size_ch0):
            if i == size_ch0-1:
                print(f"Frame {i+1} of {size_ch0}", end='\n')
            else:
                print(f"Frame {i+1} of {size_ch0}", end='\r')
            sz = image.shape
            warp_mode = cv2.MOTION_TRANSLATION
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            number_of_iterations = 5000
            termination_eps = 1e-10
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

            (cc, warp_matrix_ch0) = cv2.findTransformECC(ref, image[i,:,:], warp_matrix, warp_mode, criteria)
            aligned = cv2.warpAffine(image[i,:,:], warp_matrix_ch0, (sz[2], sz[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            frames_ch0.append(aligned)
            warp_ch0.append(warp_matrix_ch0)

            if two_ch:
                if i <= size_ch1-1:
                    (cc, warp_matrix_ch1) = cv2.findTransformECC(ref2, image2[i, :, :], warp_matrix, warp_mode, criteria)
                    aligned2 = cv2.warpAffine(image2[i,:,:], warp_matrix_ch1, (sz[2], sz[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    frames_ch1.append(aligned2)
                    warp_ch1.append(warp_matrix_ch1)
        if two_ch:
            return np.array(frames_ch0), np.array(frames_ch1), warp_ch0, warp_ch1
        else:
            return np.array(frames_ch0), None, warp_ch0, None

    def align_and_register2(self, image, image2, ref, two_ch=False):
        print("Perforing Motion Correction - deblur")
        frames_ch0 = []
        frames_ch1 = []
        warp_ch0 = []
        size_ch0 = image.shape[0]
        size_ch1 = image2.shape[0]
        for i in range(size_ch0):
            if i == size_ch0-1:
                print(f"Frame {i+1} of {size_ch0}", end='\n')
            else:
                print(f"Frame {i+1} of {size_ch0}", end='\r')
            sz = image.shape
            warp_mode = cv2.MOTION_TRANSLATION
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            number_of_iterations = 5000
            termination_eps = 1e-10
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

            (cc, warp_matrix_ch0) = cv2.findTransformECC(ref, image[i,:,:], warp_matrix, warp_mode, criteria)
            aligned = cv2.warpAffine(image[i,:,:], warp_matrix_ch0, (sz[2], sz[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            frames_ch0.append(aligned)
            warp_ch0.append(warp_matrix_ch0)

            if two_ch:
                if i <= size_ch1-1:
                    aligned2 = cv2.warpAffine(image2[i,:,:], warp_matrix_ch0, (sz[2], sz[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    frames_ch1.append(aligned2)

        if two_ch:
            return np.array(frames_ch0), np.array(frames_ch1), warp_ch0
        else:
            return np.array(frames_ch0), None, warp_ch0

    def peak_finder(self, image):
        image = image.T
        struct_disk_radius = 2
        bw_threshold_tolerance = 0.7
        img_frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        disk_size = 2 * struct_disk_radius -1
        disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk_size, disk_size))

        tophatted_frame = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, disk_kernel)

        hist_data = cv2.calcHist([tophatted_frame], [0], None, [256], [0, 256])
        hist_data[0] = 0

        peak_width, peak_location = fwhm(hist_data)
        bw_threshold = int(peak_location + bw_threshold_tolerance * peak_width)

        # Apply gaussian filter to the top-hatted image [fspecial, imfilter]
        blurred_tophatted_frame = cv2.GaussianBlur(tophatted_frame, (3, 3), 0)

        # Convert the filtered image to b/w [im2bw]
        bw_frame = cv2.threshold(
            blurred_tophatted_frame, bw_threshold, 255, cv2.THRESH_BINARY
        )[1]

        # "Open" the b/w image (in a morphological sense) [imopen]
        bw_opened = cv2.morphologyEx(
            bw_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        )

        # Fill holes ofsize 1 pixel in the resulting image [bwmorph]
        bw_filled = cv2.morphologyEx(
            bw_opened,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        spot_locations = ultimate_erode(bw_filled[:, :], image)
        if np.isnan(spot_locations).any():
            raise "Found nans"

        return spot_locations

    def peak_finder_1d(self, line):
        peaks, _ = signal.find_peaks(line, distance=40)
        return peaks

    def filter_spots(self, dims, spots):
        non_redundant = [spots[0]]
        for spot in spots:
            for keep in non_redundant:
                xmin = keep[0] - 12
                xmax = keep[0] + 12
                ymin = keep[1] - 12
                ymax = keep[1] + 12
                if spot[0]>=xmax or spot[0]<=xmin or spot[1]>=ymax or spot[1]<=ymin:
                    if keep == non_redundant[-1]:
                        non_redundant.append(spot)
                        break
                    else:
                        continue
                else:
                    break
        #Remove edge spots
        remove_edge = []
        for spot in non_redundant:
            if spot[0] <= 3 or spot[0] >= dims[0]-3:
                pass
            elif spot[1] <= 3 or spot[1] >= dims[1]-3:
                pass
            else:
                remove_edge.append(spot)
        return np.array(remove_edge)

    def subsample_frames(self):
        frames = []
        start = 150
        step = 25
        stack = self.read(self.args['input'], 142, 300)
        stack_0 = stack[:,:,428:855]
        ref = self.average(stack_0)
        for i in range(193):
            print(f'Reading image: {i}')
            im = self.read(self.args['input'], start, start+step)
            im_0 = im[:,:,428:855]
            average = self.average(im_0)
            align = self.align(average, ref)
            frames.append(align.T)
            start += step
        print("Write average.")
        self.write(np.array(frames), "aligned_frames.tiff")

    def identify_single_mols(self):
        frames = []
        all_peaks = []
        start = 150
        step = 25
        stack = self.read(self.args['input'], 142, 400)
        stack_0 = stack[:,:,428:855]
        ref = self.average(stack_0)
        for i in range(ref.shape[0]):
            line = ref[i,:]
            peaks = self.peak_finder_1d(line)
            for j in peaks:
                all_peaks.append((i, j))
        for j in range(ref.shape[1]):
            line = ref[:,j]
            peaks = self.peak_finder_1d(line)
            for i in peaks:
                all_peaks.append((i, j))
        grid = np.zeros(ref.shape)
        coodinates = np.array(all_peaks)
        new_vals = np.ones(coodinates.shape[0])
        grid[tuple(zip(*coodinates))] = new_vals
        self.write(grid.T, "peaks.tiff")

    def intensity_trace(self, stack, coord, box=2):
        padding = 2

        x_min = int(coord[0] - box)
        x_max = int(coord[0] + box)
        y_min = int(coord[1] - box)
        y_max = int(coord[1] + box)

        bg_x_min = int(x_min - padding)
        bg_x_max = int(x_max + padding + 1)
        bg_y_min = int(y_min - padding - 1)
        bg_y_max = int(y_max + padding)

        trace = []
        bg = []
        particle_stack = []

        for step in range(stack.shape[0]):
            frame = stack[step,:,:]
            ROI = frame[x_min:x_max,y_min:y_max]
            ROI_bg = frame[bg_x_min:bg_x_max,bg_y_min:bg_y_max]

            I = np.sum(ROI) / (ROI.shape[0] ** 2)
            background = (np.sum(ROI_bg) - np.sum(ROI)) / ((ROI_bg.shape[0] ** 2) - (ROI.shape[0] ** 2))

            trace.append(I)
            bg.append(background)
            particle_stack.append(ROI_bg)

        return np.array(trace), np.array(bg), particle_stack

    def smooth(self, trace, box_pts=10, type=''):
        if type=='convolve':
            box = np.ones(box_pts)/box_pts
            return np.convolve(trace, box, mode='same')
        elif type=='denoise':
            return denoise_tv_chambolle(trace, weight=0.1)
        elif type=='butter':
            B, A = signal.butter(2, 0.05)
            return signal.filtfilt(B,A,trace)
        else:
            return trace

    def normalise(self, trace, bg):
        max = np.max(trace)
        min = np.min(bg)
        return (trace - min) / (max - min)

    def plotting(self, liposome_trace, protein_trace, fit, new_fit, residuals, ignore_fit=False):
        def chunker(seq, step):
            return (seq[pos:pos+step] for pos in range(0, len(seq), step))

        def _multigaussian(x, *curves):
            sum = np.zeros(x.shape[0])
            for curve in chunker(curves, 3):
                amp, centre, sigma = curve
                func = amp*(1/sigma*(np.sqrt(2*np.pi)))*np.exp((-1.0/2.0)*(((x-centre)/sigma)**2))
                sum = np.add(sum, func)
            return sum

        # Define figure layout
        left, width = 0.05, 0.7
        bottom, height = 0.05, 0.8
        spacing = 0.005
        rect_plot = [left, bottom, width, height]
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        rect_resi = [left, bottom + height + spacing, width, 0.1]

        # Figure
        fig = plt.figure(figsize=(12, 7))

        # Define a trace plot and histogram.
        ax = fig.add_axes(rect_plot)
        ax.set_ylim([-0.5, 2])
        ax.set_xlim([-2, len(liposome_trace)+2])
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histy.tick_params(axis='y', labelleft=False)
        ax_resi = fig.add_axes(rect_resi, sharex=ax)
        ax_resi.tick_params(axis='x', labelbottom=False)

        # Feed the data to figure object
        ax.plot(liposome_trace)
        ax.plot(protein_trace)
        ax.plot(fit)
        ax.plot(new_fit)
        ax_resi.plot(residuals)

        # Generate histogram
        binwidth = 0.1
        max, min = np.max(liposome_trace), np.min(liposome_trace)
        bins = np.arange(min, max + binwidth, binwidth)
        counts, _, _ = ax_histy.hist(liposome_trace, bins=bins, orientation='horizontal')

        #Fit multiple gaussians to histogram
        peaks = set(fit)
        p0 = [[1, centre, 0.2] for centre in peaks]
        p0 = [param for sublist in p0 for param in sublist]

        if not ignore_fit:
            try:
                popt_gauss, pcov_gauss = optimize.curve_fit(_multigaussian, bins[:-1], counts, p0=p0)
            except RuntimeError:
                print("Failed to fit multi-Gaussian function. Trying again with bounds")
                low_b0 = []
                high_b0 = []
                for param in chunker(p0, 3):
                    _, lim_1, lim_2 = param
                    if lim_1 < 0:
                        lim_1_low = 1.3 * lim_1
                        lim_1_high = 0.7 * lim_1
                    else:
                        lim_1_low = 0.7 * lim_1
                        lim_1_high = 1.3 * lim_1

                    if lim_2 < 0:
                        lim_2_low = 1.5 * lim_2
                        lim_2_high = 0.5 * lim_2
                    else:
                        lim_2_low = 0.5 * lim_2
                        lim_2_high = 1.5 * lim_2

                    low_b0.append([0, lim_1_low, lim_2_low])
                    high_b0.append([100, lim_1_high, lim_2_high])

                low_b0 = [lim for sublist in low_b0 for lim in sublist]
                high_b0 = [lim for sublist in high_b0 for lim in sublist]

                popt_gauss, pcov_gauss = optimize.curve_fit(_multigaussian, bins[:-1], counts, p0=p0,
                                                            bounds=(low_b0, high_b0))
            finally:
                ax_histy.plot(_multigaussian(bins[:-1], *popt_gauss), bins[:-1], 'k--')

                for params in chunker(popt_gauss, 3):
                    gauss = _multigaussian(bins[:-1], *params)
                    ax_histy.plot(gauss, bins[:-1], c=np.random.rand(3, ))

        plt.show()
        plt.close()

    def plotting2(self, liposome_trace, fit, protein_trace, new_fit, residuals, stack1, stack2, num, ignore_fit=False, two_ch=True):
        def chunker(seq, step):
            return (seq[pos:pos+step] for pos in range(0, len(seq), step))

        def _multigaussian(x, *curves):
            sum = np.zeros(x.shape[0])
            for curve in chunker(curves, 3):
                amp, centre, sigma = curve
                func = amp*(1/sigma*(np.sqrt(2*np.pi)))*np.exp((-1.0/2.0)*(((x-centre)/sigma)**2))
                sum = np.add(sum, func)
            return sum

        # Define figure layout
        spacing = 0.05
        #LEFT, BOTTOM, RIGHT, TOP
        RRS = [0.05, 0.05, 0.95, 0.2]
        rect_plot = [0.05, 0.255, 0.7, 0.6]
        rect_histy = [0.755, 0.255, 0.2, 0.6]
        rect_resi = [0.05, 0.86, 0.7, 0.1]


        # Figure
        fig = plt.figure(figsize=(14, 7))

        # Define a trace plot and histogram.
        ax = fig.add_axes(rect_plot)
        ax.set_ylim([-0.5, 1.5])
        ax.set_xlim([-2, len(liposome_trace)+2])
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histy.tick_params(axis='y', labelleft=False)
        ax_resi = fig.add_axes(rect_resi, sharex=ax)
        ax_resi.tick_params(axis='x', labelbottom=False)

        ##########################################################################
        # step = 5
        # num_plots = int(len(stack1) / step)
        num_plots = 80
        step = int(int(len(stack1)) / num_plots)

        # # Define figure layout
        if two_ch:
            (ax_images, ax2_images) = fig.subplots(2, num_plots)
        else:
            ax_images = fig.subplots(1, num_plots)
        plt.subplots_adjust(left=RRS[0],bottom=RRS[1],right=RRS[2],top=RRS[3], hspace=spacing, wspace=2*spacing)

        for im in ax_images.flatten():
            im.set_yticklabels([])
            im.set_xticklabels([])
            im.set_yticks([])
            im.set_xticks([])

        if two_ch:
            for im2 in ax2_images.flatten():
                im2.set_yticklabels([])
                im2.set_xticklabels([])
                im2.set_yticks([])
                im2.set_xticks([])

        lipo_min, lipo_max = np.min(stack1), np.max(stack1)
        if two_ch:
            prot_min, prot_max = np.min(stack2), np.max(stack2)

        j = 0
        ax_images[0].set_ylabel(f'\u03BB488', fontstyle='italic', fontsize='small', color='royalblue')
        if two_ch:
            ax2_images[0].set_ylabel(f'\u03BB647', fontstyle='italic', fontsize='small', color='indianred')
        for i in range(num_plots):
            ax_images[i].set_title(f'{j * self.dt} s', fontstyle='italic', fontsize='small')
            ax_images[i].imshow(stack1[j], vmin=lipo_min, vmax=lipo_max)
            if two_ch:
                ax2_images[i].imshow(stack2[j], vmin=prot_min, vmax=prot_max, cmap='coolwarm')
            j = j + step
        #########################################################################

        # Feed the data to figure object
        ax.plot(liposome_trace, linewidth=2, color='royalblue')
        ax.plot(fit, '--', color='navy')
        if two_ch:
            ax.plot(protein_trace, linewidth=2, color='indianred')
            ax.plot(new_fit, '--', color='maroon')
        x = np.arange(0,len(residuals),1)
        col_vals = residuals - 1 / (-1 - 1)
        ax_resi.scatter(x, residuals, c=cm.PiYG(col_vals), edgecolor='none')

        # Generate histogram
        binwidth = 0.02
        lipo_max, lipo_min = np.max(liposome_trace), np.min(liposome_trace)
        lipo_bins = np.arange(lipo_min, lipo_max + binwidth, binwidth)
        lipo_counts, _, _ = ax_histy.hist(liposome_trace, bins=lipo_bins, orientation='horizontal', color='royalblue',
                                          alpha=0.5)

        if two_ch:
            prot_max, prot_min = np.max(protein_trace), np.min(protein_trace)
            prot_bins = np.arange(prot_min, prot_max + binwidth, binwidth)
            prot_counts, _, _ = ax_histy.hist(protein_trace, bins=prot_bins, orientation='horizontal', color='indianred',
                                              alpha=0.5)


        #Fit multiple gaussians to histogram
        peaks = set(fit)
        p0 = [[1, centre, 0.2] for centre in peaks]
        p0 = [param for sublist in p0 for param in sublist]

        if not ignore_fit:
            try:
                popt_gauss, pcov_gauss = optimize.curve_fit(_multigaussian, bins[:-1], counts, p0=p0)
            except RuntimeError:
                print("Failed to fit multi-Gaussian function. Trying again with bounds")
                low_b0 = []
                high_b0 = []
                for param in chunker(p0, 3):
                    _, lim_1, lim_2 = param
                    if lim_1 < 0:
                        lim_1_low = 1.3 * lim_1
                        lim_1_high = 0.7 * lim_1
                    else:
                        lim_1_low = 0.7 * lim_1
                        lim_1_high = 1.3 * lim_1

                    if lim_2 < 0:
                        lim_2_low = 1.5 * lim_2
                        lim_2_high = 0.5 * lim_2
                    else:
                        lim_2_low = 0.5 * lim_2
                        lim_2_high = 1.5 * lim_2

                    low_b0.append([0, lim_1_low, lim_2_low])
                    high_b0.append([100, lim_1_high, lim_2_high])

                low_b0 = [lim for sublist in low_b0 for lim in sublist]
                high_b0 = [lim for sublist in high_b0 for lim in sublist]

                popt_gauss, pcov_gauss = optimize.curve_fit(_multigaussian, bins[:-1], counts, p0=p0,
                                                            bounds=(low_b0, high_b0))
            finally:
                ax_histy.plot(_multigaussian(bins[:-1], *popt_gauss), bins[:-1], 'k--')

                for params in chunker(popt_gauss, 3):
                    gauss = _multigaussian(bins[:-1], *params)
                    ax_histy.plot(gauss, bins[:-1], c=np.random.rand(3, ))

        plt.savefig(self.args['out'] + f'SM_trace_{num}')
        # plt.show()
        plt.close()

    def plot_2d(self, liposome_stack, protein_stack):
        step = 2
        num_plots = int(len(liposome_stack) / step)
        # Define figure layout
        left, right = 0.01, 0.99
        bottom, top = 0.3, 0.7
        hspace, wspace = 0.5, 0.05

        f, axarr = plt.subplots(2, num_plots, figsize=(18,3))
        plt.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace)

        for ax in axarr.flatten():
            ax.axis('off')

        lipo_min, lipo_max = np.min(liposome_stack), np.max(liposome_stack)
        prot_min, prot_max = np.min(protein_stack), np.max(protein_stack)

        j = 0
        for i in range(num_plots):
            if i == 0:
                axarr[0, i].set_ylabel(f'\u03BB488') #, fontstyle='italic', fontsize='small')
                axarr[1, i].set_ylabel(f'\u03BB647') #, fontstyle='italic', fontsize='small')

            axarr[0, i].set_title(f'{j * self.dt} s', fontstyle='italic', fontsize='small')
            axarr[1, i].set_title(f'{j * self.dt} s', fontstyle='italic', fontsize='small')
            axarr[0, i].imshow(liposome_stack[j], vmin=lipo_min, vmax=lipo_max)
            axarr[1, i].imshow(protein_stack[j], vmin=prot_min, vmax=prot_max)
            j = j + step
        return f, axarr

    def photobleaching_rate(self, trace, stack, k, type=''):
        do_smart_divide = False

        def monoExp(x, m, t, b):
            return m * np.exp(-t * x) + b

        if self.args['do_calculate']:
            stack_ch0 = self.read(self.args['input'], full=True)
            ch0 = []
            for i in range(stack_ch0.shape[0]):
                I = np.sum(stack_ch0[i, :, :])
                ch0.append(I)

            p0 = (ch0[0], 0.1, 0)
            bins = np.arange(0, len(ch0), 1)
            params, cv = optimize.curve_fit(monoExp, bins, ch0, p0)

            plt.plot(ch0)
            plt.plot(monoExp(bins, *params))
            np.savetxt('photobleaching.txt', ch0)
            np.savetxt('exp_fit.txt', monoExp(bins, *params))
            print(f'Photobleaching params: decay constant, k = {params[1]}')

            if self.args['input2']:
                stack_ch1 = self.read(self.args['input2'], full=True)
                ch1 = []
                for i in range(stack_ch1.shape[0]):
                    I = np.sum(stack_ch1[i, :, :])
                    ch1.append(I)

                p0 = (max(ch1), 0.1, 0)
                x_max = max(range(len(ch1)), key=ch1.__getitem__)
                bins = np.arange(0, len(ch1), 1)
                params, cv = optimize.curve_fit(monoExp, bins[x_max:-1], ch1[x_max:-1], p0)
                print(f'Photobleaching params: decay constant, k = {params[1]}')

                plt.plot(ch1)
                plt.plot(monoExp(bins, *params))

            plt.show()

        else:
            if type == 'liposome':
                A0 = trace[0]
            elif type == 'protein':
                A0 = trace[25]
            else:
                print('No type...')

            bins = np.arange(0, len(trace), 1)
            fit = monoExp(bins, A0, k, 0)
            error = trace - fit
            mask = np.zeros(error.shape, dtype=bool)
            mask[np.abs(error) > 0.15] = True
            decay = np.copy(fit)
            if do_smart_divide:
                np.place(decay, mask, 1)

            corr = []
            for i, frame in enumerate(stack):
                arr = frame / fit[i]
                corr.append(arr)

            corrected_trace = trace / decay
            return corrected_trace, np.array(corr), error

    def localisations_2_coords(self, warp_matrix):
            ch0_stack = []
            ch1_stack = []
            for root, _, filenames in os.walk(self.args['localisations']):
                if os.path.split(root)[1] == 'ch0_locs':
                    l = []
                    for file in filenames:
                        l.append(int(file.split('_')[-4][1:]))
                    lsort = sorted(range(len(l)), key=lambda k: l[k])

                    for index in lsort:
                        img = os.path.join(root, filenames[index])
                        arr = self.read(img, full=True)
                        ch0_stack.append(arr)

                if os.path.split(root)[1] == 'ch1_locs':
                    l = []
                    for file in filenames:
                        l.append(int(file.split('_')[-4][1:]))
                    lsort = sorted(range(len(l)), key=lambda k: l[k])

                    for index in lsort:
                        img = os.path.join(root, filenames[index])
                        arr = self.read(img, full=True)
                        ch1_stack.append(arr)
            ch0_stack = np.array(ch0_stack)

            frames = []
            sz = ch0_stack.shape
            for i in range(sz[0]):
                aligned = cv2.warpAffine(ch0_stack[i, :, :], warp_matrix[i], (sz[2], sz[1]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                frames.append(aligned)
            aligned = np.array(frames)
            ii = np.where(aligned != 0)
            return list(zip(*ii[1:3]))

    def single_mol_trace(self):
        #ASSUMES PREPROCESSING HAS ALREADY BEEN DONE.
        # TAKES A GAIN REFERENCE CORRECTED STACK IN SINGLE CHANNEL FORMAT (currently)
        debug = True

        if self.args['input']:
            stack_0 = self.read(self.args['input'], 0, 5)
            ch0_average = self.average(stack_0)
        else:
            sys.exit("At minimum, ch0 should be provided. Ch1 is optional")

        if self.args['input2']:
            stack_1 = self.read(self.args['input2'], 0, 5)
            ch1_average = self.average(stack_1)
            two_ch = True
            print("Performing two-channel analysis")
        else:
            two_ch = False
            print("Performing single-channel analysis")

        full_stack_ch0 = self.read(self.args['input'], full=True)
        ch0_ref = np.max(full_stack_ch0, axis=0)

        if two_ch:
            full_stack_ch1 = self.read(self.args['input2'], full=True)
            ch1_ref = np.max(full_stack_ch1, axis=0)
        else:
            full_stack_ch1 = None
            ch1_ref = None

        full_stack_ch0, full_stack_ch1 = self.register(full_stack_ch0, full_stack_ch1, self.args['register'])
        full_stack_ch0, full_stack_ch1, warp_ch0 = self.align_and_register2(full_stack_ch0,
                                                                              full_stack_ch1,
                                                                              ch0_ref,
                                                                              two_ch=two_ch)

        all_spots = self.localisations_2_coords(warp_ch0)
        centers = self.filter_spots(stack_0[0, :, :].shape, all_spots)

        ch0_maxI_al = np.max(full_stack_ch0, axis=0)
        if two_ch:
            ch1_maxI_al = np.max(full_stack_ch1, axis=0)

        if debug:
            self.write(full_stack_ch0, "corrected_ch0_stack.tiff")
            self.write(ch0_maxI_al, "ch0_max_i_proj_drift.tiff")
            self.write(ch0_ref, "ch0_max_i_proj.tiff")
            self.write(ch0_average, "ch0_early_average.tiff")
            grid = np.zeros(ch0_average.shape)
            coodinates = np.array(centers)
            new_vals = np.ones(coodinates.shape[0])*100
            grid[tuple(zip(*coodinates))] = new_vals
            self.write(grid, "ch0_peaks.tiff")

            if two_ch:
                self.write(ch1_average, "ch1_early_average.tiff")
                self.write(ch1_ref, "ch1_max_i_proj.tiff")
                self.write(ch1_maxI_al, "ch1_max_i_proj_drift.tiff")
                self.write(full_stack_ch1, "corrected_ch1_stack.tiff")

        exclude = []
        for num,spot in enumerate(centers):
            lipo_trace, lipo_bg, lipo_stack = self.intensity_trace(full_stack_ch0, spot, box=3)
            # TO DO: Remove this... make sure
            if lipo_trace is None:
                exclude.append(spot)
                continue

            if two_ch:
                protein_trace, prot_bg, protein_stack = self.intensity_trace(full_stack_ch1, spot, box=3)
                if protein_trace is None:
                    exclude.append(spot)
                    continue

            lipo_trace = self.normalise(lipo_trace, bg=lipo_bg)
            if two_ch:
                protein_trace = self.normalise(protein_trace, bg=prot_bg[0])

            lipo_trace, lipo_stack, lipo_residuals = self.photobleaching_rate(lipo_trace, lipo_stack, self.liposome_decay, type='liposome')

            # if two_ch:
                #protein_trace = self.photobleaching_rate(protein_trace, protein_stack, self.protein_decay, type='protein')

            #self.plot_2d(lipo_stack, protein_stack)

            lipo_trace_denoise = self.smooth(np.array(lipo_trace), type='denoise')
            if two_ch:
                protein_trace_denoise = self.smooth(np.array(protein_trace), type='butter')

            boundary_mask = self.find_boundaries(lipo_trace)
            new_fit = self.piecewise_constant(lipo_trace, boundary_mask)

            events, jumps, fit = self.extract_intensities(lipo_trace, boundary_mask)

            # self.plotting(lipo_trace, protein_trace, lipo_trace_denoise, protein_trace_denoise, lipo_residuals ignore_fit=True)

            if two_ch:
                self.plotting2(lipo_trace, lipo_trace_denoise, protein_trace, protein_trace_denoise, lipo_residuals,
                               lipo_stack, protein_stack, num, ignore_fit=True, two_ch=two_ch)
            else:
                self.plotting2(lipo_trace, lipo_trace_denoise, None, None, lipo_residuals,
                               lipo_stack, None, num, ignore_fit=True, two_ch=two_ch)

        # grid = np.zeros(ch0_average.shape)
        # coodinates = np.array(exclude)
        # new_vals = np.ones(coodinates.shape[0]) * 100
        # grid[tuple(zip(*coodinates))] = new_vals
        # self.write(grid, "peaks_excluded.tiff")

    def multi_colour_preprocess(self):
        if self.args['do_gain_ref']:
            print("Trying to calculate gain ref")
            gain_ref = self.average(self.read(self.args['gain_ref']))
        elif not self.args['do_gain_ref']:
            print("found a gain ref")
            gain_ref = self.read(self.args['gain_ref'], full=True)
        else:
            print("No gain ref command provided")

        if self.args['batch'] is True and os.path.isdir(self.args['input']):
            ch0_stack = []
            ch1_stack = []
            for root, _, filenames in os.walk(self.args['input']):
                if os.path.split(root)[1] == 'ch0':
                    l = []
                    for file in filenames:
                        l.append(int(file.split('_')[-3][1:]))
                    lsort = sorted(range(len(l)), key=lambda k: l[k])

                    for index in lsort:
                        img = os.path.join(root, filenames[index])
                        arr = self.read(img, full=True)[:,0:427] - gain_ref[:,0:427]
                        #mean = np.average(arr)
                        #arr[arr > 5 * mean] = 1
                        ch0_stack.append(arr)
                elif os.path.split(root)[1] == 'ch1':
                    l = []
                    for file in filenames:
                        l.append(int(file.split('_')[-3][1:]))
                    lsort = sorted(range(len(l)), key=lambda k: l[k])

                    for index in lsort:
                        img = os.path.join(root, filenames[index])
                        arr = self.read(img, full=True)[:,428:855] - gain_ref[:,428:855]
                        #mean = np.average(arr)
                        #arr[arr > 5 * mean] = 1
                        ch1_stack.append(arr)

                else:
                    print(f"Skipping non-tif file(s) type in {root}")

            self.write(self.average(np.array(ch0_stack)), 'ch0_average.tiff')
            self.write(self.average(np.array(ch1_stack)), 'ch1_average.tiff')

            self.write(np.array(ch0_stack), 'ch0_stack.tiff')
            self.write(np.array(ch1_stack), 'ch1_stack.tiff')

        else:
            #Do single frame processing
            pass

    def survival_analysis(self):
        neg_stack = []
        exp_stack = []
        for root, _, filenames in os.walk(self.args['localisations']):
            if os.path.split(root)[1] == 'ch0_locs':
                l = []
                for file in filenames:
                    l.append(int(file.split('_')[-4][1:]))
                lsort = sorted(range(len(l)), key=lambda k: l[k])

                for index in lsort:
                    img = os.path.join(root, filenames[index])
                    arr = self.read(img, full=True)
                    ii = np.where(arr != 0)
                    l = list(zip(*ii[1:3]))
                    exp_stack.append(len(l))
        exp = np.array(exp_stack)

        # for root, _, filenames in os.walk(self.args['localisations2']):
        #     if os.path.split(root)[1] == 'ch0_locs':
        #         l = []
        #         for file in filenames:
        #             l.append(int(file.split('_')[-4][1:]))
        #         lsort = sorted(range(len(l)), key=lambda k: l[k])
        #
        #         for index in lsort:
        #             img = os.path.join(root, filenames[index])
        #             arr = self.read(img, full=True)
        #             jj = np.where(arr != 0)
        #             l = list(zip(*jj[1:3]))
        #             neg_stack.append(len(l))
        # neg = np.array(neg_stack)

        def normalise_list(data):
            return data / data[0]
        # neg_hat = normalise_list(neg)
        exp_hat = normalise_list(exp)
        # plt.plot(neg_hat[0:100], color='green')
        # plt.plot(exp_hat[0:100], color='blue')
        np.savetxt(str(self.args['out'])+'.txt', exp_hat)
        # np.savetxt('lipo_neg_control.txt', neg_hat)
        # plt.show()

    def register(self, ch_0, ch_1, warp_matrix):
        if self.args['tetraspeck']:
            stack = self.read(self.args['stack'], full=True)
            ch0 = stack[..., 0:427]
            ch1 = stack[..., 428:855]
            av0 = self.average(np.array(ch0))
            av1 = self.average(np.array(ch1))
            img_aligned, warp_matrix = self.align(av0, av1)
            self.write(av0, 'ch0_average.tiff')
            self.write(av1, 'ch1_average.tiff')
            self.write(img_aligned, 'ch1_aligned.tiff')
            np.save('warp_matrix.npy', warp_matrix)
        else:
            warp = np.load(warp_matrix)
            print("Perforing Register Correction - deblur")
            frames_ch1 = []
            size_ch1 = ch_1.shape[0]
            for i in range(size_ch1):
                if i == size_ch1 - 1:
                    print(f"Frame {i + 1} of {size_ch1}", end='\n')
                else:
                    print(f"Frame {i + 1} of {size_ch1}", end='\r')
                sz = ch_1.shape

                aligned = cv2.warpAffine(ch_1[i, :, :], warp, (sz[2], sz[1]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                frames_ch1.append(aligned)
            return ch_0, np.array(frames_ch1)


if __name__ == '__main__':
    # Argument parser
    ap = argparse.ArgumentParser(
        description=textwrap.dedent('''\
        ###########################################################################################################
                    TIRF analysis ***VIEW*** - python programs for analysis of single molecule TIRF
                    Developed by Charles Bayly-Jones (2022) - Monash University, Melbourne, Australia
        ###########################################################################################################'''),
        usage='Use python3 %(prog)s --help for more information',
        formatter_class=argparse.RawTextHelpFormatter)

    ap.add_argument("-i", "--input",
                    required=False,
                    help=textwrap.dedent('''\
                    Single image in .tif or .tiff format
                    e.g. photobleaching_run.tif'''))
    ap.add_argument("-i2", "--input2",
                    required=False,
                    help=textwrap.dedent('''\
                    Single image in .tif or .tiff format
                    e.g. photobleaching_run.tif'''))
    ap.add_argument("-k", "--stack",
                    required=False,
                    help=textwrap.dedent('''\
                    Single image in .tif or .tiff format
                    e.g. photobleaching_run.tif'''))
    ap.add_argument("-o", "--out",
                    required=False,
                    help=textwrap.dedent('''\
                    Output directory to store results'''))
    ap.add_argument("-b", "--batch", action='store_true',
                    required=False,
                    help=textwrap.dedent('''\
                    Path to directory for batch processing. Data should be organised /data/run/ch0/image.tif'''))
    ap.add_argument("-t", "--traces", action='store_true',
                    required=False,
                    help=textwrap.dedent('''\
                    Path to directory for batch processing. Data should be organised /data/run/ch0/image.tif'''))
    ap.add_argument("-s", "--survival", action='store_true',
                    required=False,
                    help=textwrap.dedent('''\
                    Path to directory for batch processing. Data should be organised /data/run/ch0/image.tif'''))
    ap.add_argument("-g", "--gain_ref",
                    required=False,
                    help=textwrap.dedent('''\
                    Is the image a stack of frames or individual frames organised by directory?'''))
    ap.add_argument("-l", "--localisations",
                    required=False,
                    help=textwrap.dedent('''\
                    Is the image a stack of frames or individual frames organised by directory?'''))
    ap.add_argument("-l2", "--localisations2",
                    required=False,
                    help=textwrap.dedent('''\
                    Is the image a stack of frames or individual frames organised by directory?'''))
    ap.add_argument("-z", "--zstack", action='store_true',
                    required=False,
                    help=textwrap.dedent('''\
                    Is the image a stack of frames or individual frames organised by directory?'''))
    ap.add_argument("-G", "--do_gain_ref", action='store_true',
                    required=False,
                    help=textwrap.dedent('''\
                        Is the image a stack of frames or individual frames organised by directory?'''))
    ap.add_argument("-e", "--do_calculate", action='store_true',
                    required=False,
                    help=textwrap.dedent('''\
                        Path to directory for batch processing. Data should be organised /data/run/ch0/image.tif'''))
    ap.add_argument("-r", "--tetraspeck", action='store_true',
                    required=False,
                    help=textwrap.dedent('''\
                        Path to directory for batch processing. Data should be organised /data/run/ch0/image.tif'''))
    ap.add_argument("-R", "--register",
                    required=False,
                    help=textwrap.dedent('''\
                            Path to directory for batch processing. Data should be organised /data/run/ch0/image.tif'''))

    args = vars(ap.parse_args())
    C = core(args)
    if args['do_calculate']:
        C.photobleaching_rate(None, None, None)
    if args['batch']:
        C.multi_colour_preprocess()
    if args['traces']:
        C.single_mol_trace()
    if args['survival']:
        C.survival_analysis()
    if args['tetraspeck']:
        C.register(None, None, None)
