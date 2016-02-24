import csv

from mmds.utils.plot_utils import GeoMap, COLORS


if __name__ == "__main__":
    data_file = "../../TwitterData/k_means_geo_gt_8_out"
    file_reader = csv.reader(open(data_file, "r"))
    next(file_reader)
    GeoMap().plot_points(file_reader, lambda row: COLORS[int(row[3])], lambda row:[row[0], row[2]])
