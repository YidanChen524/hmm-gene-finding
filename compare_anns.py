#
# compare_anns.py <true> <pred>
#
# compares a predicted gene structure against the true gene structure
# and computes various statistics summarizing the quality of the
# prediction. The argument <true> is the name of file containing the
# true gene structure in fasta format, and <pred> is the name of a
# file containing the predicted gene structure in fasta format, both
# files are assumed to containind only one sequence in fasta format,
# e.g.
#
# > python compare_anns.py true-ann6.fa pred-ann6.fa
# > Cs   (tp=757332, fp=164766, tn=305197, fn=57217): Sn = 0.9298, Sp = 0.8213, AC = 0.6213
# > Rs   (tp=715865, fp=127462, tn=304830, fn=57584): Sn = 0.9255, Sp = 0.8489, AC = 0.6603
# > Both (tp=1473197, fp=292228, tn=247613, fn=114801): Sn = 0.9277, Sp = 0.8345, AC = 0.4520
#
# Christian Storm <cstorm@birc.au.dk>
#

import os
import sys
import string


def read_ann(filename):
    lines = []
    for l in open(filename).readlines():
        if l[0] != ">" and l[0] != ';':
            lines.append(l.strip())
    return "".join(lines)


def count_c(true, pred):
    total = tp = fp = tn = fn = 0
    for i in range(len(true)):
        if pred[i] == 'C' or pred[i] == 'c':
            total = total + 1
            if true[i] == 'C' or true[i] == 'c':
                tp = tp + 1
            else:
                fp = fp + 1
        if pred[i] == 'N' or pred[i] == 'n':
            if true[i] == 'N' or true[i] == 'n' or true[i] == 'R' or true[i] == 'r':
                tn = tn + 1
            else:
                fn = fn + 1
    return total, tp, fp, tn, fn


def count_r(true, pred):
    total = tp = fp = tn = fn = 0
    for i in range(len(true)):
        if pred[i] == 'R' or pred[i] == 'r':
            total = total + 1
            if true[i] == 'R' or true[i] == 'r':
                tp = tp + 1
            else:
                fp = fp + 1
        if pred[i] == 'N' or pred[i] == 'n':
            if true[i] == 'N' or true[i] == 'n' or true[i] == 'C' or true[i] == 'c':
                tn = tn + 1
            else:
                fn = fn + 1
    return total, tp, fp, tn, fn


def count_cr(true, pred):
    total = tp = fp = tn = fn = 0
    for i in range(len(true)):
        if pred[i] == 'C' or pred[i] == 'c' or pred[i] == 'R' or pred[i] == 'r':
            total = total + 1
            if (pred[i] == 'C' or pred[i] == 'c') and (true[i] == 'C' or true[i] == 'c'):
                tp = tp + 1
            elif (pred[i] == 'R' or pred[i] == 'r') and (true[i] == 'R' or true[i] == 'r'):
                tp = tp + 1
            else:
                fp = fp + 1
        if pred[i] == 'N' or pred[i] == 'n':
            if true[i] == 'N' or true[i] == 'n':
                tn = tn + 1
            else:
                fn = fn + 1
    return total, tp, fp, tn, fn


def print_stats(tp, fp, tn, fn):
    sn = float(tp) / (tp + fn)
    sp = float(tp) / (tp + fp)
    acp = 0.25 * (float(tp)/(tp+fn) + float(tp)/(tp+fp) + float(tn)/(tn+fp) + float(tn)/(tn+fn))
    ac = (acp - 0.5) * 2
    print("Sn = %.4f, Sp = %.4f, AC = %.4f" % (sn, sp, ac))
    return ac


def print_all(true, pred):
    (totalc, tp, fp, tn, fn) = count_c(true, pred)
    if totalc > 0:
        print("Cs   (tp=%d, fp=%d, tn=%d, fn=%d):" % (tp, fp, tn, fn), end=" ")
        print_stats(tp, fp, tn, fn)

    (totalr, tp, fp, tn, fn) = count_r(true, pred)
    if totalr > 0:
        print("Rs   (tp=%d, fp=%d, tn=%d, fn=%d):" % (tp, fp, tn, fn), end=" ")
        print_stats(tp, fp, tn, fn)

    (total, tp, fp, tn, fn) = count_cr(true, pred)
    if totalc > 0 and totalr > 0:
        print("Both (tp=%d, fp=%d, tn=%d, fn=%d):" % (tp, fp, tn, fn), end=" ")
        ac = print_stats(tp, fp, tn, fn)
        return ac


if __name__ == '__main__':

    # Read true annotation
    true_ann = read_ann(sys.argv[1])

    # Read predicted annotations
    pred_ann = read_ann(sys.argv[2])

    # Check annoation length
    if len(true_ann) != len(pred_ann):
        print("ERROR: The lengths of two predictions are different")
        print("Expected %d, but found %d" % (len(true_ann), len(pred_ann)))
        sys.exit(1)

    # Print stats
    print_all(true_ann, pred_ann)
