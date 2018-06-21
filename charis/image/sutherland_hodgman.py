def clip(subjectPolygon, clipPolygon):
    if len(subjectPolygon) == 0 or len(clipPolygon) == 0:
        return []
    # code here super ugly, because copied from internet
    # https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    def inside(p):
        return(cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return ((n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3)
    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        if len(outputList) == 0:
            return []
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
    return outputList


def area(polygon):
    return abs(sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in segments(polygon))) / 2


def segments(polygon):
    if len(polygon) >= 1:
        return zip(polygon, polygon[1:] + [polygon[0]])
    else:
        return []
