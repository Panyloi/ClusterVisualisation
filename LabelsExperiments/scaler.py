def scale(df: dict) -> dict:
    '''scales input points

    Args
    ----
    df: input DataFrame instance

    Return
    ------

    dict: input df instance but with prescaled points
    '''

    x_points = df["x"].to_numpy()
    y_points = df["y"].to_numpy()
    points   = list(zip(x_points, y_points))

    ul_point = (min(points, key=lambda x: x[0])[0], max(points, key=lambda x: x[1])[1])
    dr_point = (max(points, key=lambda x: x[0])[0], min(points, key=lambda x: x[1])[1])

    # scaling - all values are prescaled base on the upper-left and lower-right points soo that they
    # fit a square of bounds (-100, 100). The proportions should be kept
    dx       = (dr_point[0] - ul_point[0])
    dy       = (ul_point[1] - dr_point[1])
    sunit    = None
    x_shift  = None # shift are counted from left/down
    y_shift  = None #
    
    # unit calculation based on which orientation is wider
    if dx >= dy:
        sunit = dx/200
        x_shift = 0
        y_shift = (dx - dy)/2
    else:
        sunit = dy/200
        x_shift = (dx - dy)/2
        y_shift = 0

    for row in df.iterrows():
        df.at[row[0], 'x'] = 0
        df.at[row[0], 'y'] = 0
