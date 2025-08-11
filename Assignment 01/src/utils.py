
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- IO helpers ----
def imread_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def imread_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image_rgb(path, img_rgb):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def save_image_gray(path, img_gray):
    cv2.imwrite(path, img_gray)

# ---- Plot helpers ----
def show_gray(img, title=None):
    plt.imshow(img, cmap='gray')
    if title: plt.title(title)
    plt.axis('off')
    plt.show()

def show_rgb(img, title=None):
    plt.imshow(img)
    if title: plt.title(title)
    plt.axis('off')
    plt.show()

def plot_hist(img, title=None):
    if img.ndim == 2:
        plt.hist(img.ravel(), bins=256, range=(0,255))
    else:
        for c in range(3):
            plt.hist(img[..., c].ravel(), bins=256, range=(0,255), alpha=0.5)
    if title: plt.title(title)
    plt.show()

# ---- Task 1 & 2: LUT transforms ----
def apply_lut(img_gray, lut):
    lut = np.asarray(lut).astype(np.uint8)
    return cv2.LUT(img_gray, lut)

# ---- Task 3: Gamma on L* in Lab ----
def gamma_correction_lab(img_rgb, gamma=0.5):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    L_norm = L / 255.0
    L_gamma = np.clip((L_norm ** gamma) * 255.0, 0, 255).astype(np.uint8)
    lab_gamma = cv2.merge([L_gamma, a, b])
    out = cv2.cvtColor(lab_gamma, cv2.COLOR_LAB2RGB)
    return out, L, L_gamma

# ---- Task 4: Vibrance transform on S (HSV) ----
def vibrance_transform_on_s(img_rgb, a=0.5, sigma=70):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    x = np.arange(256).astype(np.float32)
    term = a * 128.0 * np.exp(-((x - 128.0)**2) / (2.0 * sigma**2))
    lut = np.minimum(x + term, 255.0).astype(np.uint8)
    s2 = cv2.LUT(s, lut)
    hsv2 = cv2.merge([h, s2, v])
    out = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)
    return out, lut

# ---- Task 5: Histogram equalization (custom) ----
def hist_equalize_custom(img_gray):
    hist, _ = np.histogram(img_gray.flatten(), bins=256, range=(0,256))
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_norm = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_norm, 0).astype(np.uint8)
    out = cdf_final[img_gray]
    return out, hist, cdf

# ---- Task 7: Sobel filtering ----
SOBEL_X = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
SOBEL_Y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)

def conv2d(img_gray, kernel):
    kh, kw = kernel.shape
    pad_y, pad_x = kh//2, kw//2
    padded = cv2.copyMakeBorder(img_gray, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
    out = np.zeros_like(img_gray, dtype=np.float32)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            roi = padded[y:y+kh, x:x+kw]
            out[y,x] = np.sum(roi * kernel)
    return out

def sobel_manual(img_gray):
    gx = conv2d(img_gray, SOBEL_X)
    gy = conv2d(img_gray, SOBEL_Y)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag, gx, gy

def sobel_separable(img_gray):
    kx1 = np.array([1,2,1], dtype=np.float32).reshape(3,1)
    kx2 = np.array([1,0,-1], dtype=np.float32).reshape(1,3)
    # gx = (kx1 * kx2) convolved with image
    gx = conv2d(img_gray, kx1 @ kx2)
    # gy = transpose
    gy = conv2d(img_gray, (kx1 @ kx2).T)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag, gx, gy

# ---- Task 8: Zooming ----
def zoom_nearest(img, s):
    in_h, in_w = img.shape[:2]
    out_h, out_w = int(in_h * s), int(in_w * s)
    out = np.zeros((out_h, out_w) + (() if img.ndim==2 else (img.shape[2],)), dtype=img.dtype)
    for y in range(out_h):
        for x in range(out_w):
            src_y = min(int(round(y / s)), in_h-1)
            src_x = min(int(round(x / s)), in_w-1)
            out[y,x] = img[src_y, src_x]
    return out

def zoom_bilinear(img, s):
    in_h, in_w = img.shape[:2]
    out_h, out_w = int(in_h * s), int(in_w * s)
    out = np.zeros((out_h, out_w) + (() if img.ndim==2 else (img.shape[2],)), dtype=np.float32)
    for y in range(out_h):
        for x in range(out_w):
            src_y = (y + 0.5)/s - 0.5
            src_x = (x + 0.5)/s - 0.5
            y0 = int(np.floor(src_y)); x0 = int(np.floor(src_x))
            y1 = min(y0 + 1, in_h - 1); x1 = min(x0 + 1, in_w - 1)
            wy = src_y - y0; wx = src_x - x0
            y0 = np.clip(y0, 0, in_h - 1); x0 = np.clip(x0, 0, in_w - 1)
            Ia = img[y0, x0].astype(np.float32)
            Ib = img[y0, x1].astype(np.float32)
            Ic = img[y1, x0].astype(np.float32)
            Id = img[y1, x1].astype(np.float32)
            top = Ia * (1-wx) + Ib * wx
            bottom = Ic * (1-wx) + Id * wx
            out[y,x] = top * (1-wy) + bottom * wy
    return np.clip(out, 0, 255).astype(img.dtype)

def ssd_normalized(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    num = np.sum((a - b)**2)
    den = np.sum(a**2) + 1e-9
    return num / den
