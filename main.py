# https://www.kaggle.com/datasets/crawford/emnist
# https://arxiv.org/pdf/1702.05373v1

import argparse
import cv2 # pip install opencv-python
import numpy as np
import os
import pandas as pd
import pickle
import pygame
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import _utils
import _myModel

# initialize logging
logger = _utils.initializeLogging()
logger.info('Starting program')
_utils._logger = logger
_myModel._logger = logger

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', help='The random seed to be used.', type=int, default=None)
parser.add_argument('--train', help='Train the neural network against the test data.', action='store_true')
parser.add_argument('--data_file', help='The file to use for training the network.', type=str, default='emnist-balanced.csv')
parser.add_argument('--validation_size', help='The percent to reserve of our dataset for validation. (0-50)', type=int, default=20)
parser.add_argument('--testing_size', help='The percent to reserve of our dataset for testing. (0-50)', type=int, default=5)
parser.add_argument('--epochs', help='The number of epochs/iterations to train the network.', type=int, default=50)
parser.add_argument('--batch_size', help='The size of mini-batches for training.', type=int, default=64)
parser.add_argument('--read_model', help='Read the information from a saved model file.', action='store_true')
parser.add_argument('--model_file', help='The name of the model file to use.', type=str, default='model.pkl')
parser.add_argument('--show_charts', help='Show the history charts from a saved model.', action='store_true')
parser.add_argument('--show_image', help='Will show an image from the training set.', action='store_true')
parser.add_argument('--index', help='The index of the image to show.', type=int, default=0)
parser.add_argument('--create_dataset', help='Will create a dataset from a source dataset.', action='store_true')
parser.add_argument('--save_file', help='The file to save the new dataset to.', type=str, default='subset.csv')
parser.add_argument('--samples', help='The number of random samples to save in the new dataset from the source.', type=int, default=1000)
args = parser.parse_args()

### PYGAME ###
def runPygame(model):
    def printText(text, location, size = 16):
        font = pygame.font.SysFont('Courier', size)
        text = font.render(text, True, (255, 255, 255), (0, 0, 0))
        textRect = location
        screen.blit(text, textRect)

    def resetScreen(modelOutputNodes):
        # get the label mapping
        labelMap = _utils.getLabelMapping(modelOutputNodes)
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, drawingAreaBackgroundColor, drawingAreaRect)
        row = 0
        if modelOutputNodes <= 31:
            spacing = (screenWidth - 30) / (modelOutputNodes + 1)
            itemsPerRow = modelOutputNodes
        else:
            spacing = (screenWidth - 30) / ((modelOutputNodes / 2) + 1)
            itemsPerRow = ((modelOutputNodes + 1) // 2)
        topStart = 465 if modelOutputNodes > 31 else 495
        for i in range(len(labelMap)):
            row = i // itemsPerRow
            col = i % itemsPerRow
            char = labelMap[i + 1] if modelOutputNodes == 26 else labelMap[i]
            printText(char, (col * spacing + spacing + 3, topStart + (row * 80)), 24)

    def drawProbabilities(probabilities, modelOutputNodes):
        probabilities = np.squeeze(probabilities)
        if modelOutputNodes <= 31:
            pygame.draw.rect(screen, (0, 0, 0), (0, 440, screenWidth, 55))
        else:
            pygame.draw.rect(screen, (0, 0, 0), (0, 410, screenWidth, 55))
            pygame.draw.rect(screen, (0, 0, 0), (0, 490, screenWidth, 55))
        if modelOutputNodes <= 31:
            spacing = (screenWidth - 30) / (modelOutputNodes + 1)
            itemsPerRow = modelOutputNodes
        else:
            spacing = (screenWidth - 30) / ((modelOutputNodes / 2) + 1)
            itemsPerRow = ((modelOutputNodes + 1) // 2)
        topStart = 465 if modelOutputNodes > 31 else 495
        for i, prob in enumerate(probabilities):
            row = i // itemsPerRow
            col = i % itemsPerRow
            height = int(prob * 55)
            top = topStart - height + (80 * row)
            pygame.draw.rect(screen, probabilityRectColor, (col * spacing + spacing + 2, top, 15, height))

    def getImageData():
        # get the data to be assessed
        drawRect = pygame.Rect(drawingAreaRect)
        # check if we have anything to assess yet
        if drawBounds[2] <= drawBounds[0] or drawBounds[3] <= drawBounds[1]:
            return None
        buffer = 0.2
        left = drawBounds[0]
        top = drawBounds[1]
        right = drawBounds[2]
        bottom = drawBounds[3]
        width = right - left
        height = bottom - top
        if width < 50:
            left -= (50 - width) / 2
            right += (50 - width) / 2
            width = right - left
        if height < 50:
            top -= (50 - height) / 2
            bottom += (50 - height) / 2
            height = bottom - top
        if width > height * 1.2:
            top = top - ((width - height) / 2)
            bottom = bottom + ((width - height) / 2)
            height = bottom - top
        elif height > width * 1.2:
            left = left - ((height - width) / 2)
            right = right + ((height - width) / 2)
            width = right - left
        xOffset = width * buffer
        yOffset = height * buffer
        drawRect = pygame.Rect(
            max(drawBounds[0] - xOffset, drawingAreaRect[0]), 
            max(drawBounds[1] - yOffset, drawingAreaRect[1]),
            min((drawBounds[2] - drawBounds[0]) + (2 * xOffset), drawingAreaRect[2]),
            min((drawBounds[3] - drawBounds[1]) + (2 * yOffset), drawingAreaRect[3])
        )
        if drawRect[0] < 0:
            drawRect[0] = 0
        if drawRect[1] < 0:
            drawRect[1] = 0
        if drawRect[0] + drawRect[2] > drawingAreaRect[2]:
            drawRect[2] = drawingAreaRect[2] - drawRect[0]
        if drawRect[1] + drawRect[3] > drawingAreaRect[3]:
            drawRect[3] = drawingAreaRect[3] - drawRect[1]

        subsurface = screen.subsurface(drawRect)
        imageData = pygame.surfarray.array3d(subsurface)
        imageData = cv2.resize(imageData, (28, 28), interpolation=cv2.INTER_AREA)
        # reshape the data for the neural netwrk model
        imageData = np.mean(imageData, axis=-1)     # convert to grayscale
        imageData = imageData.reshape(1, 28 * 28)   # reshape to 28 x 28
        imageData = 255 - imageData                 # flip to white on black (this is what the network was trained on)
        imageData = imageData / 255.                # normalize the values
        return imageData

    def showImage(imageData, labelData, index):
        imagePixels = imageData[index, :].reshape(28, 28) * 255.0

        imagePixels = np.rot90(imagePixels, k=1)
        imagePixels = np.flipud(imagePixels)

        if labelData is not None:
            logger.info(f'Image label: {np.argmax(labelData[:, index])}')
        plt.imshow(imagePixels, cmap='gray')
        plt.axis("off")
        plt.show()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # this prevents the model predictions from logging to the console

    # load the model
    print(model.summary())
    modelInputLayer = model.layers[0].__class__.__name__
    logger.info(f'Input layer type: {modelInputLayer}')
    modelOutputNodes = model.layers[-1].units
    logger.info(f'Output layer nodes: {modelOutputNodes}')

    # initialize pygame
    pygame.init()
    screenWidth = 545
    screenHeight = 575
    screen = pygame.display.set_mode((screenWidth, screenHeight), pygame.DOUBLEBUF)
    pygame.display.set_caption('Neural Network - EMNIST')

    # variables
    drawingAreaBackgroundColor = (255, 255, 255)
    drawingAreaForegroundColor = (0, 0, 0)
    drawingAreaRect = pygame.Rect(0, 0, screenWidth-1, 400)
    drawBounds = [drawingAreaRect.width, drawingAreaRect.height, 0, 0]
    probabilityRectColor = (0, 200, 0)

    resetScreen(modelOutputNodes)

    drawing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left mouse button
                    drawing = True
                elif event.button == 3:  # right mouse button
                    resetScreen(modelOutputNodes)
                    drawBounds = [drawingAreaRect.width, drawingAreaRect.height, 0, 0]
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    x, y = event.pos
                    if drawingAreaRect.collidepoint(event.pos):
                        pygame.draw.circle(screen, drawingAreaForegroundColor, event.pos, 5)
                        drawBounds[0] = min(drawBounds[0], x)
                        drawBounds[1] = min(drawBounds[1], y)
                        drawBounds[2] = max(drawBounds[2], x)
                        drawBounds[3] = max(drawBounds[3], y)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    imageData = getImageData()
                    showImage(imageData, None, 0)

        pygame.display.update()

        # test the image
        imageData = getImageData()
        if imageData is not None:
            if modelInputLayer == 'Conv2D':
                imageData = imageData.reshape(-1, 28, 28, 1).astype("float32")  # Ensure correct shape
            probabilities = model.predict_on_batch(imageData)
            drawProbabilities(probabilities, modelOutputNodes)

def loadData(data_file):
    logger.info(f'Loading data file: {data_file}')
    csvData = pd.read_csv(f'datasets\\{data_file}', header=None)
    data = csvData.iloc[:, 1:].values
    data = data / 255.0
    labels = csvData.iloc[:, 0].values.reshape(-1, 1)
    logger.info(f'Samples: {len(data)}')

    return data, labels

def prepareData(data, labels, validation_percent=None, test_percent=10, shuffle=True, random_seed=None):
    np.random.seed(random_seed)
    logger.info('Preparing data...')

    total_samples = len(data)
    test_samples = max(int(total_samples * (test_percent / 100.0)), 1)
    validation_samples = max(int(total_samples * (validation_percent / 100.0)), 1) if validation_percent else 0

    # we want to shuffle
    if shuffle:
        logger.info('Shuffling data...')
        indices = np.random.permutation(data.shape[0])
        data = data[indices, :]
        labels = labels[indices, :]

    # validation data
    if validation_percent:
        logger.info(f'Validation samples: {validation_samples}')
        validation_data = data[:validation_samples, :]
        validation_labels = labels[:validation_samples, :]
        data = data[validation_samples:, :]
        labels = labels[validation_samples:, :]
    else:
        validation_data = None
        validation_labels = None

    # test data
    logger.info(f'Test samples: {test_samples}')
    test_data = data[:test_samples, :]
    test_labels = labels[:test_samples, :]
    data = data[test_samples:, :]
    labels = labels[test_samples:, :]

    # training data
    logger.info(f'Training samples: {len(data)}')
    train_data = data
    train_labels = labels

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def showAccuracyChart(trainingAccuracy, validationAccuracy):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cost', color='blue')
    ax1.plot(np.arange(1, len(trainingAccuracy) + 1), trainingAccuracy, color='blue', label='Cost')
    ax1.tick_params(axis='y', labelcolor='blue')
    if validationAccuracy:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (%)', color='green')
        ax2.plot(np.arange(1, len(validationAccuracy) + 1), validationAccuracy, color='green', label='Accuracy')
        ax2.tick_params(axis='y', labelcolor='green')
        plt.title('Cost and Accuracy')
    else:
        plt.title('Cost')
    plt.show()

# we want to train the network
if args.train:
    logger.info('Training the network (--train)')
    dataFile = args.data_file
    randomSeed = args.random_seed
    validationSize = args.validation_size
    testingSize = args.testing_size
    epochs = args.epochs
    batchSize = args.batch_size
    saveFile = args.model_file

    # some input validations
    if validationSize not in range(0, 51):
        logger.error('Invalid validation_size. Needs to be 0-50.')
        exit()
    if testingSize not in range(0, 51):
        logger.error('Invalid testing_size. Needs to be 0-50.')
        exit()
    if validationSize + testingSize >= 80:
        logger.error('Validation and testing subsets are too large. Needs to be less than 80.')
        exit()

    # show a message if a random seed was specified
    if randomSeed:
        logger.important(f'Random seed specified: {randomSeed}')

    # load and prepare our data
    data, labels = loadData(dataFile)
    trainX, trainY, validationX, validationY, testX, testY = prepareData(data, labels, validationSize, testingSize, randomSeed)

    # pretty basic model
    logger.info('Initializing model...')
    
    # grab our model definition
    outputNodes = trainY.max() + 1
    model, early_stopping = _myModel.getMyModel(outputNodes)

    # we need to know the input layer so we can format our data properly
    modelInputLayer = model.layers[0].__class__.__name__

    # check if we need to reshape the data for conv2d
    if modelInputLayer == 'Conv2D':
        trainX = trainX.reshape(-1, 28, 28, 1).astype("float32")
        validationX = validationX.reshape(-1, 28, 28, 1).astype("float32")
        testX = testX.reshape(-1, 28, 28, 1).astype("float32")

    # train our model
    logger.info('Training model...')
    logger.info(f'Epochs: {epochs}')
    logger.info(f'Batch size: {batchSize}')
    history = model.fit(
        trainX, trainY,
        epochs=epochs,
        batch_size=batchSize,
        validation_data=(validationX, validationY) if validationSize > 0 else None,
        callbacks=[early_stopping]
    )

    # check if we're evaluating our performance
    testLoss, testAccuracy = None, None
    if testingSize > 0:
        testLoss, testAccuracy = model.evaluate(testX, testY)
        logger.info(f'Test Accuracy: {testAccuracy:.4f}')

    # check if we want to show some charts
    if args.show_charts:
        trainingAccuracy = history.history['accuracy']
        validationAccuracy = history.history['val_accuracy'] if validationSize > 0 else None
        showAccuracyChart(trainingAccuracy, validationAccuracy)
    
    # save the model
    logger.info(f'Saving the model: {saveFile}')
    saveData = {
        'model': model,
        'history': history.history
    }
    try:
        with open(saveFile, 'wb') as f:
            pickle.dump(saveData, f)
    except Exception as e:
        logger.error(f'Failed to save model fle: {e}')

# reading a model file
elif args.read_model:
    modelFile = args.model_file
    logger.info(f'Reading model file: {modelFile}')
    try:
        with open(modelFile, 'rb') as f:
            data = pickle.load(f)
        model = data['model']
        history = data['history']
        print(model.summary())

        # check if we want to show charts
        if args.show_charts:
            trainingAccuracy = history.get('accuracy', None)
            validationAccuracy = history.get('val_accuracy', None)
            showAccuracyChart(trainingAccuracy, validationAccuracy)
    except Exception as e:
        logger.error(f'Failed to read model file: {e}')
        exit()

# showing an image
elif args.show_image:
    dataFile = args.data_file
    imageIndex = args.index
    randomSeed = args.random_seed

    # show a message if a random seed was specified
    if randomSeed:
        logger.important(f'Random seed specified: {randomSeed}')

    # load and prepare our data
    data, labels = loadData(dataFile)
    
    # show an image
    _utils.showImage(data, labels, imageIndex)

# creating a dataset
elif args.create_dataset:
    # grab some parameters
    sourceFile = args.data_file
    samples = args.samples
    randomSeed = args.random_seed
    saveFile = args.save_file

    # get the source data
    logger.info(f'Creating new dataset')
    logger.info(f'Source file: {sourceFile}')
    data = pd.read_csv(f'datasets\\{sourceFile}')
    logger.info(f'Samples: {len(data)}')

    # save a subset
    randomSeed and logger.important(f'Random seed: {randomSeed}')
    if samples >= len(data):
        logger.error('Not enough samples in source file.')
    logger.info(f'Selecting samples: {samples}')
    subset = data.sample(n=samples, random_state=randomSeed)
    logger.info(f'Saving to file: {saveFile}')
    subset.to_csv(f'datasets\\{saveFile}', index=False, header=False)

# running the pygame UI
else:
    logger.info('Running user interface via pygame')
    modelFile = args.model_file
    logger.info(f'Using model file: {modelFile}')

    # read the model file
    try:
        with open(modelFile, 'rb') as f:
            data = pickle.load(f)
        model = data['model']
    except Exception as e:
        logger.error(f'Failed to read model file: {e}')
        exit()

    # run the 'game'
    runPygame(model)
logger.info('All done')
