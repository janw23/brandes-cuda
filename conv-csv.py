# Converts graph to CSV format.

import sys

for line in sys.stdin:
    print(';'.join(line.split()))
