import logging
import matplotlib.pyplot as plt
import numpy as np

IMPORTANT_LEVEL = 21
_logger = None

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    purple = "\x1b[35;1m"
    blue = "\x1b[34;1m"
    cyan = "\x1b[1;96m"
    reset = "\x1b[0m"
    log_format = "%(levelname)s | %(asctime)s.%(msecs)03d | %(message)s"

    FORMATS = {
        logging.DEBUG: purple + log_format + reset,
        logging.INFO: blue + log_format + reset,
        IMPORTANT_LEVEL: cyan + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def important(self, message, *args, **kwargs):
    if self.isEnabledFor(IMPORTANT_LEVEL):
        self._log(IMPORTANT_LEVEL, message, args, **kwargs) # pylint: disable=protected-access

def initializeLogging(level=logging.INFO):
    logging.addLevelName(IMPORTANT_LEVEL, 'IMPORTANT')
    logging.Logger.important = important
    logger = logging.getLogger('__name__')
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    handler.setFormatter(CustomFormatter())
    logger.setLevel(level)
    return logger

def showImage(imageData, labelData, index):
    uniqueLabelsCount = len(np.unique(labelData))
    labelMap = getLabelMapping(uniqueLabelsCount)

    if imageData.shape[1] == 784:
        imagePixels = imageData[index, :].reshape(28, 28) * 255.0
    else:
        imagePixels = imageData[index, :, :, :] * 255.0
    
    imagePixels = np.rot90(imagePixels, k=1)
    imagePixels = np.flipud(imagePixels)

    if labelData is not None:
        labelIndex = int(labelData[index, 0].item())
        char = labelMap[labelIndex] if labelIndex in labelMap else '?'
        _logger.info(f'Image label: {char}')
    plt.imshow(imagePixels, cmap='gray')
    plt.axis("off")
    plt.show()

def getLabelMapping(outputNodeCount):
    if outputNodeCount == 62:
        return {
            0: '0',  1: '1',  2: '2',  3: '3',  4: '4',  5: '5',  6: '6',  7: '7',  8: '8',  9: '9',
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
            20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
            30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 
            36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
            46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
            56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
        }
    elif outputNodeCount == 47:
        return {
            0: '0',  1: '1',  2: '2',  3: '3',  4: '4',  5: '5',  6: '6',  7: '7',  8: '8',  9: '9',
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
            20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
            30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
            36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
        }
    elif outputNodeCount == 10:
        return {
            0: '0',  1: '1',  2: '2',  3: '3',  4: '4',  5: '5',  6: '6',  7: '7',  8: '8',  9: '9'
        }
    elif outputNodeCount == 26:
        return {
            1: 'A',  2: 'B',  3: 'C',  4: 'D',  5: 'E',  6: 'F',  7: 'G',  8: 'H',  9: 'I', 10: 'J',
            11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T',
            21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'
        }
    else:
        raise 'Invalid node count'