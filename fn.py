import cv2
import pytesseract

from pytesseract import Output
from collections import namedtuple
from utils import has_empty_space, has_int, has_special_char, is_num, remove_duplicate_preserve_order, to_float

THA="tha"
ENG="eng"
THA_ENG="tha+eng"
RDTFTA = ["ref", "date", "time", "from", "to", "amount"]


# TODO: remove this function
def get_img_size(bank):
    sizes = {
        "BAY": (1080, 2268),
        "BBL": (1080, 2589),
        "GOV": (1080, 2037),
        "KBANK": (984, 1402),
        "KTB": (996, 1258),
        "SCB": (1080, 2540),
        "TMB": (1500, 3100)
    }

    return sizes[bank]


# TODO: remove this function
def get_ocr_locations(bank):
    OCRLocation = namedtuple("OCRLocation", ["id", "bbox"])
    locations = {
        "BAY": [
            OCRLocation("from", (500, 310, 510, 60)),
            OCRLocation("to", (500, 525, 510, 60)),
            OCRLocation("amount", (500, 705, 400, 60)),
            OCRLocation("date", (320, 165, 205, 40)),
            OCRLocation("time", (525, 165, 145, 40)),
            OCRLocation("ref", (75, 960, 305, 60))
        ],
        "BBL": [
            OCRLocation("from", (400, 1340, 400, 50)),
            OCRLocation("to", (400, 1580, 500, 50)),
            OCRLocation("amount", (400, 1100, 320, 90)),
            OCRLocation("date", (350, 940, 220, 50)),
            OCRLocation("time", (580, 940, 150, 50)),
            OCRLocation("ref", (70, 2370, 650, 50))
        ],
        "GOV": [
            OCRLocation("from", (400, 1340, 400, 50)),
            OCRLocation("to", (400, 1580, 500, 50)),
            OCRLocation("amount", (400, 1100, 320, 90)),
            OCRLocation("date", (350, 940, 220, 50)),
            OCRLocation("time", (580, 940, 150, 50)),
            OCRLocation("ref", (70, 2370, 650, 50))
        ],
        "KBANK": [
            OCRLocation("from", (220, 240, 420, 50)),
            OCRLocation("to", (220, 550, 420, 50)),
            OCRLocation("amount", (210, 985, 320, 50)),
            OCRLocation("date", (60, 100, 195, 50)),
            OCRLocation("time", (255, 100, 160, 50)),
            OCRLocation("ref", (220, 865, 400, 50))
        ],
        "KTB": [
            OCRLocation("from", (180, 480, 500, 50)),
            OCRLocation("to", (180, 680, 500, 50)),
            OCRLocation("amount", (560, 900, 320, 50)),
            OCRLocation("date", (600, 1050, 230, 50)),
            OCRLocation("time", (865, 1050, 100, 50)),
            OCRLocation("ref", (330, 360, 350, 50))
        ],
        "SCB": [
            OCRLocation("from", (350, 670, 700, 70)),
            OCRLocation("to", (350, 880, 700, 70)),
            OCRLocation("amount", (650, 1320, 370, 70)),
            OCRLocation("date", (340, 415, 250, 70)),
            OCRLocation("time", (620, 415, 120, 70)),
            OCRLocation("ref", (355, 490, 565, 70))
        ],
        "TMB": [
            OCRLocation("from", (500, 1280, 935, 70)),
            OCRLocation("to", (500, 1665, 935, 100)),
            OCRLocation("amount", (550, 740, 320, 120)),
            OCRLocation("date", (950, 1880, 270, 90)),
            OCRLocation("time", (1220, 1880, 215, 90)),
            OCRLocation("ref", (900, 2070, 535, 90))
        ]
    }

    return locations[bank]


def convert_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def get_rois(img, p, pw, ph):
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d["level"])

    [img_h, img_w] = img.shape
    min_w = int(img_w * pw)
    max_h = int(img_h * ph)

    # find lines of text
    boxes = []
    lines = []
    for i in range(n_boxes):
        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
        if x != 0 and y != 0 and h != img.shape[0] and w > min_w and h < max_h:
            boxes.append((x, y, w, h))
            lines.append(y+h)

    unique_lines = remove_duplicate_preserve_order(lines)
    long_lines = []

    # find longest area of each line
    for l in unique_lines:
        longest = 0
        longest_coor = (0, 0, 0, 0)
        for box in boxes:
            (x, y, w, h) = box
            if abs(y + h - l) < 5:
                long = x+w
                if long > longest:
                    longest = long
                    longest_coor = box
        long_lines.append(longest_coor)

    # find region of interest
    rois = []
    for i in range(len(long_lines)):
        (x, y, w, h) = long_lines[i]

        roi = img[y-p:y+h+p, x-p:x+w+p]
        rois.append(roi)
    
    return rois


def append_orc_msg(ocr):
    msg = []
    for i in range(len(RDTFTA)):
        msg.append("{}: {}".format(RDTFTA[i], ocr[i]))
    return msg


def gov_ocr(rois):
    ref = ""
    date = ""
    time = ""
    from_ = ""
    to = ""
    amount = ""

    for i in range(len(rois)):
        # ref
        if i in [1, 2]:
            if ref == "":
                txt = pytesseract.image_to_string(rois[i], lang=ENG)
                if has_empty_space(txt):
                    refs = txt.split()
                    if has_special_char(txt):
                        for r in refs:
                            if not has_special_char(r):
                                ref = r.strip()
                    elif has_int(txt):
                        ref = refs[-1].strip()
        # date time
        if i in [2, 3, 4]:
            if date == "" and time == "":
                txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
                datetime = txt.split(" ")
                date = " ".join(datetime[:3]).strip()
                time = " ".join(datetime[3:]).strip()
                if len(date) > 15 or len(time) > 15:
                    date = ""
                    time = ""
        # from
        if i in [4, 5]:
            if from_ == "":
                txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
                if not has_special_char(txt) and not has_int(txt):
                    t = txt.split(" ")
                    if (len(t) >= 2):
                        from_ = " ".join(t[-2:]).strip()
                    else:
                        from_ = t[0].strip()
        # to
        if i in [7, 8, 9]:
            if to == "":
                txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
                if not has_special_char(txt):
                    if has_empty_space(txt.strip()):
                        names = txt.split(" ")
                        to = " ".join(names[-2:]).strip()
                    else:
                        to = txt.strip()
        # amount
        if i == len(rois) - 2 or i == len(rois) - 1:
            txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
            if has_empty_space(txt):
                txt = txt.split()
                for t in txt:
                    if is_num(t):
                        if amount == "":
                            amount = t
                        else:
                            amount = t if to_float(t) > to_float(amount) else amount

    return [ref, date, time, from_, to, amount]


def scb_ocr(rois):
    ref = ""
    date = ""
    time = ""
    from_ = ""
    to = ""
    amount = ""

    for i in range(len(rois)):
        # ref
        if i in [3]:
            if ref == "":
                txt = pytesseract.image_to_string(rois[i], lang=ENG)
                if has_special_char(txt):
                    refs = txt.split(" ")
                    for r in refs:
                        if not has_special_char(r):
                            ref = r.strip().replace(" ", "")
                elif ":" in txt:
                    idx = txt.index(":")
                    ref = "".join(txt[idx+1:]).strip()
                else:
                    ref = txt.strip().replace(" ", "")
        # date time
        if i == 2:
            txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
            datetime = txt.split("-")
            date = datetime[0].strip()
            time = datetime[1].strip()
        # from
        if i in [4, 5, 6]:
            if from_ == "":
                txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
                if not has_special_char(txt) and not has_int(txt):
                    t = txt.split(" ")
                    if (len(t) >= 2):
                        from_ = " ".join(t[-2:]).strip()
                    else:
                        from_ = t[0].strip()
        # to
        if i in [7, 8, 9]:
            txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
            if not has_special_char(txt) and not has_int(txt):
                if len(txt.split(" ")) > 2:
                    names = txt.split(" ")
                    if "@" in names:
                        at_idx = names.index("@")
                        to = " ".join(names[at_idx+1:]).strip()
                    else:
                        to = " ".join(names[-2:]).strip()
                else:
                    to = txt.strip()
        # amount
        if i == len(rois) - 4 or i == len(rois) - 7:
            txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
            if has_empty_space(txt):
                txt = txt.split()[-1]
                if is_num(txt):
                    amount = txt

    return [ref, date, time, from_, to, amount]


def tmb_ocr(rois):
    ref = ""
    date = ""
    time = ""
    from_ = ""
    to = ""
    amount = ""

    for i in range(len(rois)):
        # ref
        if i in [14, 15, 16]:
            if ref == "":
                txt = pytesseract.image_to_string(rois[i], lang=ENG)
                if has_empty_space(txt) and "/" not in txt:
                    refs = txt.split()
                    if has_special_char(txt):
                        for r in refs:
                            if not has_special_char(r):
                                ref = r.strip()
                    elif has_int(txt):
                        ref = refs[-1].strip()
        # date time
        if i in [13]:
            if date == "" and time == "":
                txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
                datetime = txt.strip().split()
                date = datetime[-2]
                time = datetime[-1]
        # from
        if i in [6, 7]:
            if from_ == "":
                txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
                if not has_special_char(txt) and not has_int(txt):
                    t = txt.split(" ")
                    if (len(t) >= 2):
                        from_ = " ".join(t[-2:]).strip()
                    else:
                        from_ = t[0].strip()
        # to
        if i in [9, 10, 11]:
            if to == "":
                txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
                if not has_special_char(txt) and "-" not in txt:
                    if has_empty_space(txt.strip()):
                        names = txt.split(" ")
                        to = " ".join(names[-2:]).strip()
                    else:
                        to = txt.strip()
        # amount
        if i in [1]:
            txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
            if has_empty_space(txt):
                txt = txt.split()
                for t in txt:
                    if is_num(t):
                        if amount == "":
                            amount = t
                        else:
                            amount = t if to_float(t) > to_float(amount) else amount

    return [ref, date, time, from_, to, amount]


def ktb_ocr(rois):
    ref = ""
    date = ""
    time = ""
    from_ = ""
    to = ""
    amount = ""

    for i in range(len(rois)):
        # ref
        if i in [1]:
            if ref == "":
                txt = pytesseract.image_to_string(rois[i], lang=ENG)
                if has_empty_space(txt) and "/" not in txt:
                    refs = txt.split()
                    if has_special_char(txt):
                        for r in refs:
                            if not has_special_char(r):
                                ref = r.strip()
                    elif has_int(txt):
                        ref = refs[-1].strip()
        # date time
        if i == len(rois) - 1 or i == len(rois) - 2 or i == len(rois) - 3:
            if date == "" and time == "":
                txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
                if "-" in txt:
                    datetime = txt.strip().split("-")
                    dates = datetime[0].split()
                    date = " ".join(dates[1:])
                    time = datetime[1].strip()
        # from
        if i in [2]:
            if from_ == "":
                txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
                if not has_special_char(txt) and not has_int(txt):
                    t = txt.split(" ")
                    if (len(t) >= 2):
                        from_ = " ".join(t[-2:]).strip()
                    else:
                        from_ = t[0].strip()
        # to
        if i in [3, 4, 5]:
            if to == "":
                txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
                if not has_special_char(txt) and "-" not in txt:
                    if has_empty_space(txt.strip()):
                        names = txt.split(" ")
                        to = " ".join(names[-2:]).strip()
                    else:
                        to = txt.strip()
        # amount
        if i in [6, 7]:
            txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
            if has_empty_space(txt):
                txt = txt.split()
                for t in txt:
                    if is_num(t):
                        if amount == "":
                            amount = t
                        else:
                            amount = t if to_float(t) > to_float(amount) else amount

    return [ref, date, time, from_, to, amount]


def bbl_ocr(rois):
    ref = ""
    date = ""
    time = ""
    from_ = ""
    to = ""
    amount = ""

    for i in range(len(rois)):
        # ref
        if i == len(rois) - 2 or i == len(rois) - 4:
            if ref == "":
                txt = pytesseract.image_to_string(rois[i], lang="eng")
                if has_empty_space(txt):
                    refs = txt.split()
                    if has_special_char(txt):
                        for r in refs:
                            if not has_special_char(r) and is_num(r):
                                ref = r.strip()
        # date time
        if i in [2]:
            if date == "" and time == "":
                txt = pytesseract.image_to_string(rois[i], lang="tha+eng")
                if "," in txt:
                    datetime = txt.strip().split(",")
                    date = datetime[0].strip()
                    time = datetime[1].strip()
        # from
        if i in [4]:
            if from_ == "":
                txt = pytesseract.image_to_string(rois[i], lang="tha+eng")
                txt = txt.split("\n\n")[0]
                if not has_special_char(txt) and not has_int(txt):
                    t = txt.split(" ")
                    if (len(t) >= 2):
                        from_ = " ".join(t[-2:]).strip()
                    else:
                        from_ = t[0].strip()
        # to
        if i in [7]:
            if to == "":
                txt = pytesseract.image_to_string(rois[i], lang="tha+eng")
                txt = txt.split("\n\n")[0]
                if not has_special_char(txt) and "-" not in txt:
                    if has_empty_space(txt.strip()):
                        names = txt.split(" ")
                        to = " ".join(names[-2:]).strip()
                    else:
                        to = txt.strip()
        # amount
        if i in [3]:
            txt = pytesseract.image_to_string(rois[i], lang="tha+eng")
            if has_empty_space(txt):
                txt = txt.split()
                for t in txt:
                    if is_num(t):
                        if amount == "":
                            amount = t
                        else:
                            amount = t if to_float(t) > to_float(amount) else amount

    return [ref, date, time, from_, to, amount]


def bay_ocr(rois):
    ref = ""
    date = ""
    time = ""
    from_ = ""
    to = ""
    amount = ""

    for i in range(len(rois)):
        # ref
        if i in [0] or i == len(rois) - 1:
            if ref == "":
                txt = pytesseract.image_to_string(rois[i], lang=ENG)
                txt = txt.strip()
                if "BAY" in txt and has_int(txt):
                    ref = txt
        # date time
        if i in [1]:
            if date == "" and time == "":
                txt = pytesseract.image_to_string(rois[i], lang=THA_ENG)
                datetimes = txt.strip().split("\n")[-1]
                datetime = datetimes.split()
                date = " ".join(datetime[:3]).strip()
                time = " ".join(datetime[3:]).strip()
                if len(date) > 15 or len(time) > 15:
                    date = ""
                    time = ""
        # from
        if i in [2, 3]:
            if from_ == "":
                txt = pytesseract.image_to_string(rois[i], lang=ENG)
                if "\n" in txt:
                    txt = txt.split("\n")[0]
                if not has_special_char(txt) and not has_int(txt):
                    t = txt.split(" ")
                    if (len(t) >= 2):
                        from_ = " ".join(t[-2:]).strip()
                    else:
                        from_ = t[0].strip()
        # to
        if i in [4, 5]:
            if to == "":
                txt = pytesseract.image_to_string(rois[i], lang=ENG)
                if "\n" in txt:
                    txt = txt.split("\n")[0]
                if not has_special_char(txt):
                    to = txt.strip()
        # amount
        if i == len(rois) - 2 or i == len(rois) - 4:
            txt = pytesseract.image_to_string(rois[i], lang=ENG)
            if has_empty_space(txt):
                txt = txt.split()
                for t in txt:
                    if is_num(t):
                        if amount == "":
                            amount = t
                        else:
                            amount = t if to_float(t) > to_float(amount) else amount

    return [ref, date, time, from_, to, amount]

def kbank_ocr(rois):
    ref = ""
    date = ""
    time = ""
    from_ = ""
    to = ""
    amount = ""

    for i in range(len(rois)):
        # ref
        if i in [8, 9]:
            if ref == "":
                txt = pytesseract.image_to_string(rois[i], lang="eng")
                txt = txt.strip()
                if is_num(txt):
                    ref = txt
        # date time
        if i in [1]:
            if date == "" and time == "":
                txt = pytesseract.image_to_string(rois[i], lang="tha+eng")
                datetimes = txt.strip()
                datetime = []
                for d in datetimes:
                    if "+" not in d:
                        datetime.append(d)
                date = " ".join(datetime[:3]).strip()
                time = " ".join(datetime[3:]).strip()
                if len(date) > 15 or len(time) > 15:
                    date = ""
                    time = ""
        # from
        if i in [2]:
            if from_ == "":
                txt = pytesseract.image_to_string(rois[i], lang="tha+eng")
                if not has_special_char(txt) and not has_int(txt):
                    t = txt.split(" ")
                    if (len(t) >= 2):
                        from_ = " ".join(t[-2:]).strip()
                    else:
                        from_ = t[0].strip()
        # to
        if i in [4, 6]:
            if to == "":
                txt = pytesseract.image_to_string(rois[i], lang="tha+eng")
                if "XXX" not in txt:
                    to = txt.strip()
        # amount
        if i == len(rois) - 3 or i == len(rois) - 4:
            txt = pytesseract.image_to_string(rois[i], lang="eng")
            if has_empty_space(txt):
                txt = txt.split()
                for t in txt:
                    if is_num(t):
                        if amount == "":
                            amount = t
                        elif "," in t:
                            amount = t
                        else:
                            amount = t if to_float(t) > to_float(amount) else amount

    return [ref, date, time, from_, to, amount]
