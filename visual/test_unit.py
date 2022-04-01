from visual.DataExtractor import Extractor
from visual.Visualizer import VisBoard


def main():
    repo = Extractor()
    vis = VisBoard(repo)
    print(vis.map_int("h"))


if __name__ == '__main__':
    repo = Extractor()
    vis = VisBoard(repo)
    vis.single_var_dist("h")
