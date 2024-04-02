from enum import Enum
import itertools


class STSCColors(Enum):
    Green = (0, 150/255, 130/255)
    Blue = (70/255, 100/255, 170/255)
    PaleGreen = (130/255, 190/255, 60/255)
    Yellow = (250/255, 230/255, 20/255)
    Orange = (220/255, 160/255, 30/255)
    Brown = (160/255, 130/255, 50/255)
    Red = (160/255, 30/255, 40/255)
    Magenta = (160/255, 0, 120/255)
    Cyan = (80/255, 170/255, 230/255)

    @classmethod
    def color_list(cls):
        return [cls.Red,
                cls.Green,
                cls.Blue,
                cls.Magenta,
                cls.Orange,
                cls.Cyan,
                cls.Yellow,
                cls.PaleGreen]

    @classmethod
    def color_iterator(cls):
        colors = cls.color_list()
        for index in itertools.count():
            yield colors[index % len(colors)]
