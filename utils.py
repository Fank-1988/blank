import re

SPECIAL_CHARACTERS = "!#$%^&*?=,<>/|"


def has_special_char(string):
  return any(c in SPECIAL_CHARACTERS for c in string)


def has_int(string):
    return any(char.isdigit() for char in string)


def remove_int(string):
    return re.sub(r"\d+", "", string)


def is_num(string):
    try:
        float(string.strip().replace(",", ""))
        return True
    except:
        return False


def remove_duplicate_preserve_order(l):
  res = []
  for x in l:
    if x not in res:
      res.append(x)
  return res


def has_empty_space(string):
  return len(string.split()) > 0


def to_float(string):
    return float(string.replace(",", ""))