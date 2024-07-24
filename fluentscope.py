"""
# FluentScope
### ~ An oscilloscope that makes listening waveform more fun ~ 
(c) 2024 src3453, released under The MIT Licence. https://opensource.org/license/mit

Requirements: 
- pygame
- pyaudio
- scipy
- numpy
"""

# -*- coding:utf-8 -*-
import pygame
from pygame.locals import *
import sys
import pyaudio
import numpy as np
from scipy import signal
import time
import colorsys


p = pyaudio.PyAudio()
# set prams
try:
    INPUT_DEVICE_INDEX = int(sys.argv[1])
except IndexError:
    INPUT_DEVICE_INDEX = 0
    print(f"Warning: Input has not selected. Fallback to #0.")

pygame.init() # init Pygame

###############################################################################################
# CONFIGURATION CONSTANTS                                                                     #
# This section configures various configurations of FluentScope, you can change these in need.#
###############################################################################################

CHUNK = 2 ** 12 # Total length of waveform ring buffer, 2^10 ~ 2^14 is recommended.
BUFFER_LEN = 2 ** 3 # Division of captured waveform fragments, 2^2 ~ 2^4 is recommended. Too fast settings causes incorrect waveform!
FORMAT = pyaudio.paInt16 # PCM recording format. normally pyaudio.paInt16.
CHANNELS = 2 # Mic input count, normally 2.
RATE = 48000 # Mic sample rate, normally 44100 or 48000.
WIDTH = 960 # Screen width.
HEIGHT = 540 # Screen height.
DECIMATION = 1 # Waveform decimation for avoid lag.     
ZOOM_FACTOR = 8 # Zoom for slicing waveform and avoid waveform incontinuous point. For debugging only, PLEASE DON'T CHANGE IN NORMAL USE!
ZOOM_FACTOR_DISP = 2 # Zoom for avoid waveform incontinuous point. For debugging only, PLEASE DON'T CHANGE IN NORMAL USE!
ZOOM_FACTOR_DISP_2 = 1 # Horizontal zoom level of waveform.
WINDOW_LEN = 256 # Waveform length for input of max correration point searching algorithm. For debugging only, PLEASE DON'T CHANGE IN NORMAL USE!
WINDOW_DECIMATION = WINDOW_LEN//256 # Waveform decimation for input of max correration point searching algorithm. For debugging only, PLEASE DON'T CHANGE IN NORMAL USE!
FINAL_SMOOTHING_FACTOR = 1.3 # Factor of time-domain final smoothing, may causes incorrect waveforms in some situations (ex. chiptunes). 1.0 equals to disabled. Recommended value is 1.0 ~ 1.5.
ADVANCED_TRIGGERING = True # Enables advanced triggering, MASSIVELY RECOMMENDED IN NORMAL USE CASES!
DISABLE_TRIGGERING = False # WARNING! Disables all of triggering, for debugging only.
ENBALE_LOWPASS_FILTER = True # Enables Butterworth Low-Pass Filter for avoid noise sensation.
F_CUTOFF = 0.25 # The cutoff frequency of Butterworth LPF.
FILTER_ORDER = 6 # The numbers of order of Butterworth LPF.
PREVIEW_FILTERED_WAVEFORM = False # Show low-pass filtered waveform instead of unfiltered waveform.
AGC_DECAY_FACTOR = 1.001 # Auto Gain Control factor decay rate, normally 1.00 ~ 1.05
AGC_TARGET_GAIN = 0.75 # Target amplitude for auto gain control.
USE_ANOTHER_AGC_METHOD = False # Uses simple smoothing instead of limiter-based smoothing. Not recommended in normal use cases because it may causes over level.
CORR_WEIGHT_FACTOR = 5.0 # How important absolute position between center to fragments of correration candicates to evaluation of triggering score?
FINAL_OFFSET_CORRECTION = False # Offsets waveform by max position of waveform. Causes smoothless frames in some situations, not recommended.
OFFSET_AS_CORR_OFFSET = True # Recommended. Evaluate max correration point as final correction offset.
ENABLE_ANOTHER_OFFSET_CORRECTION = True # Not recommended in normal use cases.
CORRERATE_CANDICATION_THRESHOLD = 0.05 # How long can far away from the maximum value possible to cosidered as correration candicates?
MINIMUM_MAX_VALUE = 32 # Minimum value of max value of waveform.
DEBUG_VIEW = True # For debugging only. Enables debug view that shows triggering status and more.
SHOW_FPS = True # Show Frame rate.
LINE_THICKNESS = 2# Waveform line thickness. 
LINE_COLOR = (255,255,255)
LINE_INTERPOLATION = True # Waveform line interpolation method. 
LINE_AA = False # Enables waveform line anti-aliasing. Supports thickness, but extremely slow!
LINE_AA_SCALE = 0.5
ENABLE_FULLSCREEN = False # Enables exclusive fullscreen.
ENABLE_VSYNC = True # Enables VSync, makes frames more smoother. But it also can causes lags when insufficient performance.
ENABLE_WAVE_OFFSET_BY_FRAME_RING_COUNT = True # Enables waveform offset by time for smoother waveforms.
MAX_CORRERATION_CANDICATIONS = 256 # Max count of candication of calculation of correration score.
ADAPTIVE_SMOOTHING_IGNORE_THRESHOLD = 0.4 # How long can far away from the smoothed value before resetting value into current value?
ENABLE_VOLUME_BAR = True
VOLUME_BAR_SMOOTHING_FACTOR = 1.5
ENABLE_DC_OFFSET_CORRECTION = False
DC_OFFSET_CORRECTION_DECAY_FACTOR = 16.0
DC_OFFSET_CORRECTION_DEADZONE = 0.05
OFFSET_ZERO_LINE = True
DRAW_ZERO_LINE = True
DRAW_CENTER_LINE = True
DRAW_V_SCALE_LINE = True
DRAW_H_SCALE_LINE = True
ENABLE_SCALE_LABEL = True
SCALE_COUNT = 11
ENABLE_PITCH_DETECTION = True
PITCH_DETECTION_THRESHOLD = 10.0
PITCH_SMOOTHING_FACTOR = 1.5
PITCH_INDICATOR_MAX_FREQ = 1024
CHANGE_LINE_COLOR_BY_PITCH = True
DEBUG_VIEW_FFT = False

window = signal.windows.blackman(CHUNK)
screen = pygame.display.set_mode(
    (WIDTH, HEIGHT),
    (
        SRCALPHA | HWSURFACE | DOUBLEBUF | RESIZABLE | SCALED | (FULLSCREEN
        if ENABLE_FULLSCREEN
        else 0)
    ),
    vsync=ENABLE_VSYNC,
)
pygame.event.set_allowed([QUIT, KEYDOWN])
pygame.display.set_caption("FluentScope")
print(f"Input has been selected to #{INPUT_DEVICE_INDEX}.")
print(
    f"{CHUNK} samples ({CHUNK/RATE} s) divives into {BUFFER_LEN} fragments, {CHUNK//BUFFER_LEN} samples ({CHUNK//BUFFER_LEN/RATE} s) per each fragments."
)  # タイトルバーに表示する文字


def to_db(x, base=1):
    y = 20 * np.log10(x / base)
    return y


text = ""
old = 0
x = old
auto = False
fftclip = False
fftmax = 20000
fftmin = 0
r3p = 0
colormapi = 8
lvol, rvol, rlvol, rrvol = 0,0,0,0
font16 = pygame.font.Font(None, 8)

wave = np.zeros(CHUNK * 1)
wave_orig = np.zeros(CHUNK * 1)
start = time.time()

theta = 0
mx = 0
nmx = 0
sample_frame = 0
fftfreq = 0
old_frag = None
final = np.zeros(CHUNK // ZOOM_FACTOR_DISP)
old_final = np.zeros(CHUNK // ZOOM_FACTOR_DISP)
filt = signal.butter(
    FILTER_ORDER, F_CUTOFF, btype="low", analog=False, output="sos", fs=None
)
captured = 0
mx_old = 0
dcoffset = 0

########################################################################################
# Audio callback function that grabs waveform fragments, and transfer into ring buffer.#
########################################################################################


def callback(wavedata, frame_count, time_info, status):
    global wave, wave_orig, start, theta, mx, captured, lvol, rvol, rlvol, rrvol, dcoffset, nmx, fftfreq

    if wavedata != None:
        input = wavedata
    else:
        raise Exception("No wavedata streamed!")

    # input = stream.read(CHUNK, exception_on_overflow=False)
    # print(type(input))
    # bufferからndarrayに変換
    ndarrayl = np.frombuffer(input, dtype="int16")[0::2]
    ndarrayr = np.frombuffer(input, dtype="int16")[1::2]
    # f = np.fft.fft(ndarray)

    # ndarrayからリストに変換
    # Pythonネイティブのint型にして扱いやすくする

    # mx2=max(a)
    # print(mx2)

    # 試しに0番目に入っているものを表示してみる

    wavel = np.array(ndarrayl).astype(int)
    waver = np.array(ndarrayr).astype(int)

    #rmsl = np.sqrt(np.mean(np.power(wavel,2)))
    #rmsr = np.sqrt(np.mean(np.power(waver,2)))

    rlvol = float(np.max(np.abs(wavel)))/32768
    rrvol = float(np.max(np.abs(waver)))/32768
    # val=(rmsr-rmsl)/(rmsr+rmsl)*90
    # x += (val-old)/2
    # old=x
    # lu=np.ceil((rmsr+rmsl)/2)

    # r = x * np.pi/180
    # r2 = (rmsr-rmsl)/(rmsr+rmsl)*90 * np.pi/180
    wave_orig = np.append(wave_orig, (wavel + waver) / 2)
    wave_orig = np.delete(wave_orig, np.s_[0 : CHUNK // BUFFER_LEN])
    # print((wave_orig))
    # try:
    # snr = (max(wave[wave >= 0])-max(-1*wave[wave <= 0]))/32768
    # except ZeroDivisionError:
    # snr = 0
    # except ValueError:
    # snr = 0
    rx = 0
    _mx = max([max(wave_orig), abs(min(wave_orig)), MINIMUM_MAX_VALUE])
    if USE_ANOTHER_AGC_METHOD:
      mx += (_mx-mx)*(AGC_DECAY_FACTOR-1)
    else:
      if _mx > mx:
          mx = _mx
      else:
          mx /= AGC_DECAY_FACTOR
    mx_old = mx
    nmx = (mx+mx_old)/2
    # mx=8192
    # fft = np.fft.fft(wave*window, n=CHUNK)
    # fft = np.log10(np.fft.fft(wave*window, n=CHUNK))*10
    wave = (0.0 + wave_orig / 32768 * (32768 / (nmx)) / (1)) + 0 * 2.0
    if ENABLE_DC_OFFSET_CORRECTION:
      dcoffset += (np.mean(wave) - dcoffset) / DC_OFFSET_CORRECTION_DECAY_FACTOR
      if np.abs(np.mean(wave)) < DC_OFFSET_CORRECTION_DEADZONE:
        dcoffset = 0
    wave -= dcoffset
    captured = time.time()
    return (None, pyaudio.paContinue)


def trigger(data, start, Tlevel):
    data = data - Tlevel

    zero_crossings = np.where(np.diff(np.sign(np.roll(data, -start))) >= 1)[0]

    if len(zero_crossings) == 0:
        return -1
    else:
        return zero_crossings[0] + 1


def pad(array, length):
    if length > len(array):
        return np.pad(array, (0, max(length - len(array), 0)))
    else:
        return array

clock = pygame.time.Clock()


def render():
    global theta, old_frag, final, old_final, filt, wave, sample_frame, captured, lvol, rvol, dcoffset, nmx, fftfreq
    screen.fill((0, 0, 0))
    lvol += (rlvol - lvol) / VOLUME_BAR_SMOOTHING_FACTOR
    rvol += (rrvol - rvol) / VOLUME_BAR_SMOOTHING_FACTOR
    if not auto:
        # rx = (np.where(np.logical_and(wave < threshold,np.diff(wave,append=1) > 0))[0][0])
        # rx = (trigger(wave[CHUNK//2-CHUNK//ZOOM_FACTOR:CHUNK//2+CHUNK//ZOOM_FACTOR],CHUNK//2,np.average(wave)))+(CHUNK//2-CHUNK//ZOOM_FACTOR)
        sample_frame = int((time.time() - captured) * RATE) % CHUNK//BUFFER_LEN
        #print(sample_frame)
        # rx = np.argmax(wave)
        roll_offset = -((sample_frame
            - int(CHUNK // BUFFER_LEN * 1))
            * int(ENABLE_WAVE_OFFSET_BY_FRAME_RING_COUNT))
        #roll_offset = 0
        shifted_wave = np.roll(
            wave,
            roll_offset
        )
        #print(roll_offset)
        cut_wave = shifted_wave[
            CHUNK // 2 - CHUNK // ZOOM_FACTOR : CHUNK // 2 + CHUNK // ZOOM_FACTOR
        ]
        if ENBALE_LOWPASS_FILTER:
            cut_wave = signal.sosfiltfilt(filt, cut_wave)
        # cut_wave = (cut_wave + np.roll(cut_wave,1)) / 2
        # rx = int(np.argmax(cut_wave))
        if ADVANCED_TRIGGERING:
            """if old_frag is None:
                old_frag = cut_wave
            #print(old_frag)
            corr = np.correlate(cut_wave, old_frag, mode='full')
            old_frag = cut_wave
            rx = int(np.argmax(corr))
            print(rx)"""

            samples = []
            raw_samples = []
            samples_offset = []
            indices = np.where(
                cut_wave
                >= np.max(cut_wave)
                - (np.max(cut_wave) - np.min(cut_wave))
                * CORRERATE_CANDICATION_THRESHOLD
            )[0][:MAX_CORRERATION_CANDICATIONS]
            if old_frag is None:
                old_frag = pad(
                    shifted_wave[
                        np.argmax(cut_wave)
                        + (CHUNK // 2 - CHUNK // ZOOM_FACTOR)
                        - WINDOW_LEN // 2 : np.argmax(cut_wave)
                        + (CHUNK // 2 - CHUNK // ZOOM_FACTOR)
                        + WINDOW_LEN // 2 : WINDOW_DECIMATION
                    ],
                    WINDOW_LEN // WINDOW_DECIMATION,
                )
            for i in indices:
                try:
                    corr = np.correlate(
                        pad(
                            shifted_wave[
                                i
                                + (CHUNK // 2 - CHUNK // ZOOM_FACTOR)
                                - WINDOW_LEN // 2 : i
                                + (CHUNK // 2 - CHUNK // ZOOM_FACTOR)
                                + WINDOW_LEN // 2 : WINDOW_DECIMATION
                            ],
                            WINDOW_LEN // WINDOW_DECIMATION,
                        ),
                        old_frag,
                        "same",
                    )
                    res = np.max(corr) + (i / len(cut_wave)) * CORR_WEIGHT_FACTOR
                except IndexError:
                    res = -1
                # print(res)
                raw_samples.append(corr)
                samples.append(res)
                samples_offset.append(
                    (np.argmax(corr) - len(corr) // 2) * WINDOW_DECIMATION
                )
            # rx = (np.where(cut_wave > np.max(cut_wave)-0.01)[0][0])
            peak = np.argmax(samples)
            # print(len(samples))
            rx = indices[peak]
            # print(np.argmax(samples),rx)
            old_frag = pad(
                shifted_wave[
                    rx
                    + (CHUNK // 2 - CHUNK // ZOOM_FACTOR)
                    - WINDOW_LEN // 2 : rx
                    + (CHUNK // 2 - CHUNK // ZOOM_FACTOR)
                    + WINDOW_LEN // 2 : WINDOW_DECIMATION
                ],
                WINDOW_LEN // WINDOW_DECIMATION,
            )
            if OFFSET_AS_CORR_OFFSET:
                final_offset = samples_offset[peak]
            else:
                final_offset = np.argmax(old_frag) - len(old_frag) // 2
            if (
                ( np.max(cut_wave) - shifted_wave[rx + (CHUNK // 2 - CHUNK // ZOOM_FACTOR)] > CORRERATE_CANDICATION_THRESHOLD
                or len(indices) < 1 )
                and ENABLE_ANOTHER_OFFSET_CORRECTION
            ):
                # rx = int(np.argmax(cut_wave))
                #print("fallback")
                final_offset = np.argmax(old_frag) - len(old_frag) // 2
            if FINAL_OFFSET_CORRECTION:
                rx = rx + final_offset * WINDOW_DECIMATION
            if DEBUG_VIEW:
                points = []
                points2 = []
                maxval = max([abs(max(raw_samples[peak])), abs(min(raw_samples[peak])),0.01])
                for ji in range(len(old_frag) - 1):
                    i = int(ji)
                    i2 = int(ji) + 1
                    point1 = (i, 32 - old_frag[i] * 32)
                    point2 = (i2, 32 - old_frag[i2] * 32)
                    points.append(point1)
                    points.append(point2)
                    # print(raw_samples[peak])
                    try:
                        point1 = (
                            i + WINDOW_LEN // WINDOW_DECIMATION,
                            32 - raw_samples[peak][i] / maxval * 32,
                        )
                        point2 = (
                            i2 + WINDOW_LEN // WINDOW_DECIMATION,
                            32 - raw_samples[peak][i2] / maxval * 32,
                        )
                    except IndexError:
                        point1 = (i + WINDOW_LEN // WINDOW_DECIMATION, 0)
                        point2 = (i2 + WINDOW_LEN // WINDOW_DECIMATION, 0)
                    points2.append(point1)
                    points2.append(point2)

                # Draw lines
                pygame.draw.lines(screen, (0, 255, 0), False, points, 2)
                pygame.draw.lines(screen, (255, 0, 255), False, points2, 2)
                pygame.draw.line(
                    screen,
                    (255, 0, 0),
                    (
                        WINDOW_LEN // WINDOW_DECIMATION
                        + len(corr) // 2
                        + samples_offset[peak] // WINDOW_DECIMATION,
                        0,
                    ),
                    (
                        WINDOW_LEN // WINDOW_DECIMATION
                        + len(corr) // 2
                        + samples_offset[peak] // WINDOW_DECIMATION,
                        64,
                    ),
                    2,
                )
                pygame.draw.line(
                    screen,
                    (255, 255, 0),
                    (len(corr) // 2 + final_offset // WINDOW_DECIMATION, 0),
                    (len(corr) // 2 + final_offset // WINDOW_DECIMATION, 64),
                    2,
                )
                # for i in indices:
                # pygame.draw.line(screen, (0, 0, 255), (i-rx+(WIDTH//2),HEIGHT), (i-rx+(WIDTH//2),HEIGHT-64), 2)            #print(np.where(cut_wave > np.max(cut_wave)-0.01)[0])
                for ii, i in enumerate(samples):
                    pygame.draw.line(
                        screen,
                        (0, int(i * 256) % 256, 255),
                        (len(corr) * 2 + ii, 0),
                        (len(corr) * 2 + ii, i * 32 % 64),
                        1,
                    )
                # pygame.draw.line(screen, (255, 0, 255), (WIDTH,10), (WIDTH-(sample_frame%(CHUNK//BUFFER_LEN)*int(ENABLE_WAVE_OFFSET_BY_FRAME_RING_COUNT)),10), 3)
            # rx = np.argmax(np.diff(wave[0:CHUNK//ZOOM_FACTOR]))
            """if ADVANCED:
            samples = []
            indices = np.where(cut_wave >= np.max(cut_wave)-0.1)[0]
            if len(indices) < 2:
                indices = [int(np.argmax(cut_wave))]
            
            for i in indices:
                try:
                    res = pad(wave[i+(CHUNK//2-CHUNK//ZOOM_FACTOR)-WINDOW_LEN//2:i+(CHUNK//2-CHUNK//ZOOM_FACTOR)+WINDOW_LEN//2],0)
                    res = np.sum(np.clip(np.diff(res),0,1))
                except IndexError:
                    res = 0
                #print(res)
                samples.append(res)
            #rx = (np.where(cut_wave > np.max(cut_wave)-0.01)[0][0])
            peak = np.argmax(samples)
            #print(samples)
            rx = (indices[peak])
            if len(indices)<1:
                rx = int(np.argmax(cut_wave))
                print("fallback")
            #print(np.argmax(samples),rx)
            #print(np.where(cut_wave > np.max(cut_wave)-0.01)[0])
            #rx = np.argmax(np.diff(wave[0:CHUNK//ZOOM_FACTOR]))"""
        else:
            rx = int(np.argmax(cut_wave))

        rx = rx + (CHUNK - CHUNK // ZOOM_FACTOR)# + (CHUNK // ZOOM_FACTOR * 4)
        # rx = (CHUNK//2+trigger(wave,peak,np.average(wave))*-1)
        # wave=corr
        # print(rx)
        # progress(-1*rx,width*height,"Synced:{}".format(rx))
    # text=str(-1*rx)
    if PREVIEW_FILTERED_WAVEFORM:
        shifted_wave = signal.sosfiltfilt(filt, shifted_wave)
    _final = shifted_wave
    if DISABLE_TRIGGERING:
        rx = CHUNK
    # final = np.roll(final,int(rx))
    final2 = np.zeros(CHUNK // ZOOM_FACTOR_DISP)
    for j, i in enumerate(_final):
        try:
            final2[j - rx + int(CHUNK / ZOOM_FACTOR_DISP * 1.5)] = i
        except IndexError:
            pass
    final += (final2 - final) / FINAL_SMOOTHING_FACTOR
    final = np.where(np.abs(final2-final) > ADAPTIVE_SMOOTHING_IGNORE_THRESHOLD, final2, final)
    # final = _final
    hloffset = 0
    if OFFSET_ZERO_LINE:
      hloffset = dcoffset * HEIGHT / 2 * AGC_TARGET_GAIN
    if DRAW_ZERO_LINE:
      pygame.draw.line(screen, (64, 64, 64), (0, HEIGHT / 2 + hloffset), (WIDTH, HEIGHT / 2 + hloffset))
    if DRAW_CENTER_LINE:
      pygame.draw.line(screen, (64, 64, 64), (WIDTH / 2, 0), (WIDTH / 2, HEIGHT))
    if ENABLE_SCALE_LABEL:
      font = pygame.font.SysFont("monospace", 12, bold=True)
    if DRAW_V_SCALE_LINE:
      for y in np.linspace(-2**int(np.log2(max(nmx,1)/32768*(1/AGC_TARGET_GAIN))-0.5),2**int(np.log2(max(nmx,1)/32768*(1/AGC_TARGET_GAIN))-0.5),int((SCALE_COUNT-1))+1,endpoint=True):
        ry = (y*HEIGHT/2)*(32768/max(nmx,1))+HEIGHT/2
        if abs(y*2) <= 1.0:
          #print(ry)
          pygame.draw.line(screen, (64, 64, 64), (WIDTH / 2 - WIDTH / 2 * 0.01, ry + hloffset), (WIDTH / 2 + WIDTH / 2 * 0.01, ry + hloffset))
          if ENABLE_SCALE_LABEL and y != 0:
            screen.blit(font.render(str(format(-y*2,"+0.3f")),True,(64,64,64)),(WIDTH / 2 + WIDTH / 2 * 0.015,ry + hloffset - 6))
      for y in range(-1,1,2):
        ry = (y*HEIGHT/2)*(32768/max(nmx,1))+HEIGHT/2
        #print(ry)
        pygame.draw.line(screen, (64, 64, 64), (WIDTH / 2 - WIDTH / 2 * 0.02, ry + hloffset), (WIDTH / 2 + WIDTH / 2 * 0.02, ry + hloffset))
        if ENABLE_SCALE_LABEL and y != 0:
            screen.blit(font.render(str(format(-y*2,"+0.3f")),True,(64,64,64)),(WIDTH / 2 + WIDTH / 2 * 0.015,ry + hloffset - 6))
    if DRAW_H_SCALE_LINE:
      pass
      #for x in np.linspace(-2**int(np.log2(max(nmx,1)/32768*(1/AGC_TARGET_GAIN))-0.5),2**int(np.log2(max(nmx,1)/32768*(1/AGC_TARGET_GAIN))-0.5),int((SCALE_COUNT-1))+1,endpoint=True):
    if ENABLE_VOLUME_BAR:
        pygame.draw.rect(screen, (192, 192, 192), ((WIDTH / 2 - (WIDTH / 2 * lvol) + 1, HEIGHT-8, (WIDTH / 2 * lvol), 8)))
        pygame.draw.rect(screen, (192, 192, 192), ((WIDTH / 2 , HEIGHT-8, (WIDTH / 2 * rvol), 8)))
    if ENABLE_PITCH_DETECTION:
      fftd = np.fft.fft(shifted_wave*window).real
      try:
        rpeak,_ = signal.find_peaks(fftd,PITCH_DETECTION_THRESHOLD,PITCH_DETECTION_THRESHOLD,10)
        peak = rpeak[0]
      except IndexError:
        peak = 0
      #print(peak)
      fftfreq += ((RATE*np.fft.fftfreq(len(fftd))[peak]/(PITCH_INDICATOR_MAX_FREQ/WIDTH))-fftfreq)/PITCH_SMOOTHING_FACTOR
      #print(fftfreq)
      if peak != 0:
        pygame.draw.rect(screen, (192, 192, 192), ((fftfreq, HEIGHT-16, 10, 8)))
    if DEBUG_VIEW_FFT:
        points = []
        maxval = peak
        for ji in range(WIDTH//2 - 1):
            i = int(ji)
            i2 = int(ji) + 1
            point1 = (i, HEIGHT - fftd[i] * 1)
            point2 = (i2, HEIGHT - fftd[i2] * 1)
            points.append(point1)
            points.append(point2)
        # Draw lines
        pygame.draw.lines(screen, (128, 128, 128), False, points, 2)
        for i in rpeak:
            pygame.draw.line(screen, (255, 0, 255), (i,HEIGHT),(i,HEIGHT-16), 2)
    indices = np.linspace(
        int(WIDTH / 2 - WIDTH / ZOOM_FACTOR_DISP_2 / 2),
        int(WIDTH / 2 + WIDTH / ZOOM_FACTOR_DISP_2 / 2),
        int(WIDTH / DECIMATION),
    )

    # Prepare points for drawing lines
    points = []
    for ji, j in enumerate(list(indices)):
        i = int(j)
        i2 = int(indices[min(ji + 1, WIDTH // DECIMATION - 1)])
        point1 = (
            (i - WIDTH / ZOOM_FACTOR_DISP_2 * max(ZOOM_FACTOR_DISP_2 * 0.5 - 0.5, 0))
            * ZOOM_FACTOR_DISP_2,
            (
                final[
                    np.clip(
                        int((i) * (CHUNK / WIDTH / ZOOM_FACTOR_DISP)),
                        0,
                        CHUNK // ZOOM_FACTOR_DISP - 1,
                    )
                ]
            )
            * -HEIGHT
            / 2
            * AGC_TARGET_GAIN
            + HEIGHT / 2,
        )
        point2 = (
            (i2 - WIDTH / ZOOM_FACTOR_DISP_2 * max(ZOOM_FACTOR_DISP_2 * 0.5 - 0.5, 0))
            * ZOOM_FACTOR_DISP_2,
            (
                final[
                    np.clip(
                        int((i2 if LINE_INTERPOLATION else i) * (CHUNK / WIDTH / ZOOM_FACTOR_DISP)),
                        0,
                        CHUNK // ZOOM_FACTOR_DISP - 1,
                    )
                ]
            )
            * -HEIGHT
            / 2
            * AGC_TARGET_GAIN
            + HEIGHT / 2,
        )
        points.append(point1)
        points.append(point2)
    if CHANGE_LINE_COLOR_BY_PITCH and peak != 0:
      try:
        line_color = [int(min(max(i*255,0),255)) for i in colorsys.hsv_to_rgb(np.log2(fftfreq)/4, 0.4, 1)]
      except ValueError:
        line_color = LINE_COLOR
    else:
      line_color = LINE_COLOR
    #print(line_color)
    # Draw lines
    if LINE_AA:
        for y in range(-LINE_THICKNESS//2,LINE_THICKNESS//2,1):
            for x in range(-LINE_THICKNESS//2,LINE_THICKNESS//2,1):
                pygame.draw.aalines(screen, line_color, False, [[i[0]+x*LINE_AA_SCALE,i[1]+y*LINE_AA_SCALE] for i in points])
    else:
        pygame.draw.lines(screen, line_color, False, points, LINE_THICKNESS)
    
    # fft2 = cv2.applyColorMap(cv2.resize(np.clip(np.sqrt(np.power(fft[0:sprproc.shape[0]].real,2) + np.power(fft[0:sprproc.shape[0]].imag,2)) * scale_value*5.12,0,255).astype(np.uint8),dsize=None,fx=1,fy=2,interpolation=FFTINTER),fftcolormap)

    # texttime = font16.render(f"{'{:.1f}'.format((end*1000)):>4}ms({'{:.0f}'.format((1/end)):>4}fps) OH:{'{:.1f}'.format((end*1000-(CHUNK/RATE*1000))):>5}ms ({(CHUNK/RATE)*1000:.1f}ms bufferT)", True, (255,255,255))
    if SHOW_FPS:
        font = pygame.font.SysFont("monospace", 12, bold=True)
        # font30 = pygame.font.Font(r"cv2c\sound\16x10.ttf", 20)
        fps = font.render(str(int(clock.get_fps())), 0, (255, 255, 255))
        screen.blit(fps, (0, 0))

    # real_screen.blit(pygame.transform.smoothscale(screen,(WIDTH*2,HEIGHT*2)),(0,0),(0,0,WIDTH*2,HEIGHT*2))
    # for i in range(len(wave)):
    # pygame.draw.line(screen,color,(left[i]/80+600,-1*right[i]/80+200),(left[i]/80+600,-1*right[i]/80+200),3)

    pygame.display.update()  # 画面を更新
    # pygame.time.delay(5)
    # イベント処理
    for event in pygame.event.get():
        # print(pygame.event.get())
        if event.type == QUIT or (
            event.type == KEYDOWN and event.key == K_ESCAPE
        ):  # 閉じるボタンが押されたら終了
            pygame.quit()
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Stop Streaming")  # Pygameの終了(画面閉じられる)
            sys.exit()
    

    clock.tick(9999)


if __name__ == "__main__":
    stream = p.open(
        format=pyaudio.paInt16,
        channels=2,
        rate=RATE,
        frames_per_buffer=CHUNK // BUFFER_LEN,
        input=True,
        output=False,
        stream_callback=callback,
        input_device_index=INPUT_DEVICE_INDEX,
    )
    print("""     ______              __  ____                 
    / __/ /_ _____ ___  / /_/ __/______  ___  ___ 
   / _// / // / -_) _ \/ __/\ \/ __/ _ \/ _ \/ -_)
  /_/ /_/\_,_/\__/_//_/\__/___/\__/\___/ .__/\__/ 
                                      /_/         
  ~ An oscilloscope that makes listening waveform more fun ~
  (c) 2024 src3453 Released under The MIT License.""")
    while True:
        render()