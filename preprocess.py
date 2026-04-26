from typing import Tuple, List

import numpy as np

def induce_clipping(audio: np.ndarray)->np.ndarray:
  # Set clipping percentage to 5%
  n = int(len(audio) / 20)

  # Induce Clipping
  abs_audio = abs(audio)
  top_samples = abs_audio[np.argsort(abs_audio)[-n:]]
  threshold = top_samples[0]
  clipped_audio = np.clip(audio, -threshold, threshold)
  return clipped_audio


def clip_detection(clipped_audio: np.ndarray)->List[Tuple[int, int]]:
  #Clipping Detection Pass
  upper_max = max(clipped_audio)
  lower_min = min(clipped_audio)

  is_event = False
  start_index = -1
  end_index = -1
  grace_counter = 0
  grace_threshold = 3
  h_clip_threshold = upper_max * 0.995
  l_clip_threshold = lower_min * 0.995
  clip_events = []

  for index, sample in enumerate(clipped_audio):
    if is_event:
      if sample >= h_clip_threshold or sample <= l_clip_threshold:
        end_index = index
        grace_counter = 0
      elif grace_counter < grace_threshold:
        grace_counter += 1
      else:
        is_event = False
        clip_events.append((start_index, end_index))
    else:
      if (sample >= h_clip_threshold or sample <= l_clip_threshold) and index < len(clipped_audio)-1:
        if clipped_audio[index+1] >= h_clip_threshold or clipped_audio[index+1] <= l_clip_threshold:
          is_event = True
          start_index = index

  return clip_events

  # clip_total = 0
  # for index, event in enumerate(clip_events):
  #   clip_total += (event[1] - event[0]) + 1
  #   print(f"clip values {index}: {clip_events[index]} {clipped_audio[clip_events[index][0]]}, {clipped_audio[clip_events[index][1]]}")
  # print(f"clip percentage: {clip_total/len(clipped_audio)}")
  #
  # print(len(clip_events))
