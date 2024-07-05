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
WINDOW_LEN = 512 # Waveform length for input of max correration point searching algorithm. For debugging only, PLEASE DON'T CHANGE IN NORMAL USE!
WINDOW_DECIMATION = WINDOW_LEN//128 # Waveform decimation for input of max correration point searching algorithm. For debugging only, PLEASE DON'T CHANGE IN NORMAL USE!
FINAL_SMOOTHING_FACTOR = 1.2 # Factor of time-domain final smoothing, may causes incorrect waveforms in some situations (ex. chiptunes). Recommended value is 1.0 ~ 1.5.
ADVANCED_TRIGGERING = True # Enables advanced triggering, MASSIVELY RECOMMENDED IN NORMAL USE CASES!
DISABLE_TRIGGERING = False # WARNING! Disables all of triggering, for debugging only.
ENBALE_LOWPASS_FILTER = False # Enables Butterworth Low-Pass Filter for avoid noise sensation.
F_CUTOFF = 0.1 # The cutoff frequency of Butterworth LPF.
FILTER_ORDER = 8 # The numbers of order of Butterworth LPF.
PREVIEW_FILTERED_WAVEFORM = False # Show low-pass filtered waveform instead of unfiltered waveform.
AGC_DECAY_FACTOR = 1.005 # Auto Gain Control factor decay rate, normally 1.00 ~ 1.05
CORR_WEIGHT_FACTOR = 10.0 # How important absolute position between center to fragments of correration candicates to evaluation of triggering score?
FINAL_OFFSET_CORRECTION = False # Offsets waveform by max position of waveform. Causes smoothless frames in some situations, not recommended.
OFFSET_AS_CORR_OFFSET = False # Recommended. Evaluate max correration point as final correction offset.
CORRERATE_CANDICATION_THRESHOLD = 0.05 # How long can far away from the maximum value possible to cosidered as correration candicates?
DEBUG_VIEW = False # For debugging only. Enables debug view that shows triggering status and more.
SHOW_FPS = True # Show Frame rate.
LINE_THICKNESS = 2 # Waveform line thickness. 
ENABLE_FULLSCREEN = True # Enables exclusive fullscreen.
ENABLE_VSYNC = True # Enables VSync, makes frames more smoother. But it also can causes lags when insufficient performance.
ENABLE_WAVE_OFFSET_BY_FRAME_RING_COUNT = True # Enables waveform offset by time for smoother waveforms.

window = signal.windows.blackman(CHUNK)
screen = pygame.display.set_mode(
    (WIDTH, HEIGHT),
    (
        SRCALPHA | HWSURFACE | DOUBLEBUF | RESIZABLE | SCALED | FULLSCREEN
        if ENABLE_FULLSCREEN
        else 0
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
font16 = pygame.font.Font(None, 8)

wave = np.zeros(CHUNK * 1)
wave_orig = np.zeros(CHUNK * 1)
start = time.time()

theta = 0
mx = 0
sample_frame = 0
old_frag = None
final = np.zeros(CHUNK // ZOOM_FACTOR_DISP)
old_final = np.zeros(CHUNK // ZOOM_FACTOR_DISP)
filt = signal.butter(
    FILTER_ORDER, F_CUTOFF, btype="low", analog=False, output="sos", fs=None
)

########################################################################################
# Audio callback function that grabs waveform fragments, and transfer into ring buffer.#
########################################################################################


def callback(wavedata, frame_count, time_info, status):
    global wave, wave_orig, start, theta, mx

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
    left = [np.array(i) for i in ndarrayl]
    right = [np.array(i) for i in ndarrayr]

    # mx2=max(a)
    # print(mx2)

    # 試しに0番目に入っているものを表示してみる

    l2 = [int(s) for s in left]
    r2 = [int(s) for s in right]

    wavel = np.array(l2)
    waver = np.array(r2)

    # rmsl = np.sqrt(np.mean([elm * elm for elm in wavel]))
    # rmsr = np.sqrt(np.mean([elm * elm for elm in waver]))

    # db = to_db(rms)
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
    _mx = max([max(wave_orig), abs(min(wave_orig)), 32])
    if _mx > mx:
        mx = _mx
    else:
        mx /= AGC_DECAY_FACTOR
    # mx=8192
    # fft = np.fft.fft(wave*window, n=CHUNK)
    # fft = np.log10(np.fft.fft(wave*window, n=CHUNK))*10

    wave = (0.01 + wave_orig / 32768 * (32768 / (mx)) / (1)) + 0 * 2.0
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
    global theta, old_frag, final, old_final, filt, wave, sample_frame
    screen.fill((0, 0, 0))

    if not auto:
        # rx = (np.where(np.logical_and(wave < threshold,np.diff(wave,append=1) > 0))[0][0])
        # rx = (trigger(wave[CHUNK//2-CHUNK//ZOOM_FACTOR:CHUNK//2+CHUNK//ZOOM_FACTOR],CHUNK//2,np.average(wave)))+(CHUNK//2-CHUNK//ZOOM_FACTOR)

        # rx = np.argmax(wave)
        wave = np.roll(
            wave,
            -sample_frame
            % (CHUNK // BUFFER_LEN)
            * int(ENABLE_WAVE_OFFSET_BY_FRAME_RING_COUNT),
        )
        cut_wave = wave[
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
            )[0][:128]
            if old_frag is None:
                old_frag = pad(
                    wave[
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
                            wave[
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
            if (
                np.max(cut_wave) - wave[rx + (CHUNK // 2 - CHUNK // ZOOM_FACTOR)] > 0.2
                or len(indices) < 1
            ):
                # rx = int(np.argmax(cut_wave))
                # print("fallback")
                pass
            # print(np.argmax(samples),rx)
            old_frag = pad(
                wave[
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
            if FINAL_OFFSET_CORRECTION:
                rx = rx + final_offset * WINDOW_DECIMATION
            if DEBUG_VIEW:
                points = []
                points2 = []
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
                            32 - raw_samples[peak][i] * 2,
                        )
                        point2 = (
                            i2 + WINDOW_LEN // WINDOW_DECIMATION,
                            32 - raw_samples[peak][i2] * 2,
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

        rx = rx + (CHUNK // 2 - CHUNK // ZOOM_FACTOR) + (CHUNK // ZOOM_FACTOR * 4)
        # rx = (CHUNK//2+trigger(wave,peak,np.average(wave))*-1)
        # wave=corr
        # print(rx)
        # progress(-1*rx,width*height,"Synced:{}".format(rx))
    # text=str(-1*rx)
    if PREVIEW_FILTERED_WAVEFORM:
        wave = signal.sosfiltfilt(filt, wave)
    _final = wave
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
    # final = _final
    pygame.draw.line(screen, (64, 64, 64), (0, HEIGHT / 2), (WIDTH, HEIGHT / 2))
    pygame.draw.line(screen, (64, 64, 64), (WIDTH / 2, 0), (WIDTH / 2, HEIGHT))
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
            * 0.9
            + HEIGHT / 2,
        )
        point2 = (
            (i2 - WIDTH / ZOOM_FACTOR_DISP_2 * max(ZOOM_FACTOR_DISP_2 * 0.5 - 0.5, 0))
            * ZOOM_FACTOR_DISP_2,
            (
                final[
                    np.clip(
                        int((i2) * (CHUNK / WIDTH / ZOOM_FACTOR_DISP)),
                        0,
                        CHUNK // ZOOM_FACTOR_DISP - 1,
                    )
                ]
            )
            * -HEIGHT
            / 2
            * 0.9
            + HEIGHT / 2,
        )
        points.append(point1)
        points.append(point2)

    # Draw lines
    pygame.draw.lines(screen, (255, 255, 255), False, points, LINE_THICKNESS)

    # fft2 = cv2.applyColorMap(cv2.resize(np.clip(np.sqrt(np.power(fft[0:sprproc.shape[0]].real,2) + np.power(fft[0:sprproc.shape[0]].imag,2)) * scale_value*5.12,0,255).astype(np.uint8),dsize=None,fx=1,fy=2,interpolation=FFTINTER),fftcolormap)

    # texttime = font16.render(f"{'{:.1f}'.format((end*1000)):>4}ms({'{:.0f}'.format((1/end)):>4}fps) OH:{'{:.1f}'.format((end*1000-(CHUNK/RATE*1000))):>5}ms ({(CHUNK/RATE)*1000:.1f}ms bufferT)", True, (255,255,255))
    if SHOW_FPS:
        font = pygame.font.Font(None, 15)
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
    sample_frame = int((time.time() - start) * RATE)

    clock.tick(120)


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
    while True:
        render()
