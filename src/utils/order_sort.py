import numpy as np

def sort_boxes_tblr(boxes, y_tol=8):
    # boxes: list of (x,y,w,h)
    centers = [(x + w/2, y + h/2) for x,y,w,h in boxes]
    idx = list(range(len(boxes)))
    # group rows by y within tolerance
    rows = []
    used = set()
    for i in np.argsort([c[1] for c in centers]):
        if i in used: 
            continue
        row = [i]
        y0 = centers[i][1]
        used.add(i)
        for j in np.argsort([c[0] for c in centers]):
            if j in used: 
                continue
            if abs(centers[j][1] - y0) <= y_tol:
                row.append(j)
                used.add(j)
        row_sorted = sorted(row, key=lambda k: centers[k][0])
        rows.append(row_sorted)
    flat = [k for row in rows for k in row]
    return flat
