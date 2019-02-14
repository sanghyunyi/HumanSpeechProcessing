SEGMENTS_OFFSETS = (
    (0.00, 0.00),
    (886.00, 0.00),
    (1752.08, 0.08),  # third segment's start
    (2612.16, 0.16),
    (3572.20, 0.20),
    (4480.28, 0.28),
    (5342.36, 0.36),
    (6410.44, 0.44),  # last segment's start
    (7086.00, 0.00))  # movie's last time point

# dictionaries with paired touples containing time (2sec steps) and offset
# in respect to the audiovisual movie (forrestgump_researchcut_ger_mono.mkv)
AUDIO_AV_OFFSETS = {
    0: {  0:  21.33},
    1: {  0:  37.33,
        408:  21.33},
    2: {  0:  69.33,
        199:  61.33},
    3: {  0:  93.33,
        320: 101.33},
    4: {  0: 109.33,
        401: 101.33},
    5: {  0: 141.33},
    6: {  0: 189.31,
         61: 181.31},
    7: {  0: 205.33}}

AUDIO_AO_OFFSETS = {
    0: {  0:  47.02},
    1: {  0:  36.35,
        203:  47.02},
    2: {  0:  87.02,
        199:  92.35},
    3: {  0: 124.35,
        320: 132.35},
    4: {  0: 105.69,
        401:  92.35},
    5: {  0: 137.69,
        364: 167.02},
    6: {  0: 201.67,
         61: 543.00},
    7: {  0:-1422.31}}
