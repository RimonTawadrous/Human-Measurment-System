import math

def eclidian_distance(x1, y1, x2, y2):
    return math.sqrt(((x1-x2)**2)+((y1-y2)**2))

if __name__ == "__main__":
    print(eclidian_distance(280,324, 372,318))
    print(eclidian_distance(280,324, 372,318)*0.409)

    print(eclidian_distance(325,340, 308,421))
    print(eclidian_distance(325,340, 308,421)*0.409)

    print(eclidian_distance(308,421, 313,511))
    print(eclidian_distance(308,421, 313,511)*0.409)