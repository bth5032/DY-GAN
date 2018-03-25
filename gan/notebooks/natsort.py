#!/usr/bin/env python

import sys
import re

def natsort(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

if __name__ == "__main__":
    items = []
    for item in sys.stdin:
        items.append(item.strip())
    items = sorted(items, key=natsort)
    for item in items:
        print item


