import numpy as np
from skimage import io

def gain_ref_per_frame(stack, gain_ref):
    cache = []
    for i in range(stack.shape[0]):
        arr = stack[i,:,:] - gain_ref
        cache.append(arr)
    return np.array(cache, dtype=np.float32)

def read(image, start=0, stop=0, full=False):
    if not full:
        cache = []
        for i in range(start, stop):
            arr = io.imread(image, img_num=i)
            cache.append(arr)
        return np.array(cache, dtype=np.float32)
    else:
        return np.array(io.imread(image), dtype=np.float32)

def write(image, name):
    io.imsave(name, image)

def average(image, gain_ref=None):
    if gain_ref is not None:
        return np.average(image, axis=0) - gain_ref
    else:
        return np.average(image, axis=0)
    
def piecewise_constant(trace, boundaries):
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

def find_boundaries(trace, window=25, threshold=2):
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

def extract_intensities(trace, step_function):
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

def align(image, ref):
    sz = image.shape
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 5000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    (cc, warp_matrix) = cv2.findTransformECC(ref, image, warp_matrix, warp_mode, criteria)
    aligned = cv2.warpAffine(image, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return aligned

def align_and_register(image, image2, ref, ref2, two_ch=False):
    print("Perforing Motion Correction - deblur")
    frames_ch0 = []
    frames_ch1 = []
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

        (cc, warp_matrix) = cv2.findTransformECC(ref, image[i,:,:], warp_matrix, warp_mode, criteria)
        aligned = cv2.warpAffine(image[i,:,:], warp_matrix, (sz[2], sz[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        frames_ch0.append(aligned)

        if two_ch:
            if i <= size_ch1-1:
                (cc, warp_matrix) = cv2.findTransformECC(ref2, image2[i, :, :], warp_matrix, warp_mode, criteria)
                aligned2 = cv2.warpAffine(image2[i,:,:], warp_matrix, (sz[2], sz[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                frames_ch1.append(aligned2)
    if two_ch:
        return np.array(frames_ch0), np.array(frames_ch1)
    else:
        return np.array(frames_ch0), None


def peak_finder(image):
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

def peak_finder_1d(line):
    peaks, _ = signal.find_peaks(line, distance=40)
    return peaks

def filter_spots(dims, spots):
    non_redundant = [spots[0]]
    for spot in spots:
        for keep in non_redundant:
            xmin = keep[0] - 6
            xmax = keep[0] + 6
            ymin = keep[1] - 6
            ymax = keep[1] + 6
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

def subsample_frames():
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

def identify_single_mols():
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

def intensity_trace(stack, coord, box=2):
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

def smooth(trace, box_pts=10, type=''):
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

def normalise(trace, bg):
    max = np.max(trace)
    min = np.min(bg)
    return (trace - min) / (max - min)

def plotting(liposome_trace, protein_trace, fit, new_fit, residuals, ignore_fit=False):
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

def plotting2(liposome_trace, fit, protein_trace, new_fit, residuals, stack1, stack2, num, ignore_fit=False, two_ch=True):
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
    num_plots = 20
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

def plot_2d(liposome_stack, protein_stack):
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