'''
读入数独图片并解数独，把解出的结果贴在数独图片上
若图上的数独无解，将尝试修改题目来获得一个解
solve_sudoku()参数：
    model_path: 用于识别数字的模型的路径
    imame_path: 数独图片的路径
    show_details: 若为True, 将会展示从图中寻找数独 -> OCR结果 -> 数独求解的具体过程
    debug: 查看一系列中间处理结果
'''

# import the necessary packages
from tensorflow.python.ops.math_ops import truediv
from Utilities import extract_digit
from Utilities import find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from my_sudoku import Sudoku
import numpy as np
import imutils
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

model_path = './model/my_net_98.h5'
image_path = './test1/1-5.jpg'


def solve_sudoku(model_path, image_path, debug=False, show_details = False):
    # load the model
    print("[INFO] loading digit classifier...")
    model = load_model(model_path)

    # load the image
    print("[INFO] processing image...")
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)

    # find the puzzle in the image
    (puzzleImage, warped) = find_puzzle(image, False)
    if show_details:
        cv2.imshow("Sudoku Puzzle", puzzleImage)
        cv2.waitKey(0)

    # initialize our 9x9 sudoku board
    board = np.zeros((9, 9), dtype="int")

    # infer the location of each cell by dividing the warped image
    # into a 9x9 grid
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9

    # initialize a list to store the (x, y)-coordinates of each cell
    cellLocs = []
    # store the array of digits in the puzzle
    process = []
    # store the prediction of digits
    original = []

    # loop over the grid locations
    for y in range(0, 9):
        # initialize the current list of cell locations
        row = []
        for x in range(0, 9):
            # compute the starting and ending (x, y)
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            # crop the cell from the warped transform image and then
            # extract the digit from the cell
            cell = warped[startY+8:endY-4, startX+8:endX-4]
            digit = extract_digit(cell, False)

            # verify that the digit is not empty
            if digit is not None:
                cell = cv2.resize(cell, (28, 28))

                # resize the cell to 28x28 pixels and then prepare the
                # cell for classification
                foo = np.hstack([cell, digit])

                roi = tf.cast(digit, dtype=tf.float32)
                roi = roi / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # classify the digit and update the sudoku board with the
                # prediction
                pred = model.predict(roi).argmax(axis=1)[0]

                if pred >= 10:
                    pred = pred - 9

                original.append(pred)
                board[y, x] = pred
                process.append(foo)
            # add the (x, y)-coordinates to our cell locations list
            if digit is not None:
                row.append((startX, startY, endX, endY, True))
            else:
                row.append((startX, startY, endX, endY, False))
        # add the row to our cell locations
        cellLocs.append(row)

    if debug:
        for i in range(36):
            if i >= len(process):
                break
            plt.subplot(6, 6, i+1)
            plt.title(original[i], y=-0.7)
            plt.imshow(process[i], cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.show()

    # construct a sudoku puzzle from the board
    print("[INFO] OCR'd sudoku board:")
    puzzle = Sudoku(board=board.tolist())
    puzzle.show()

    # show the OCR result of the sudoku image
    for (cellRow, boardRow) in zip(cellLocs, puzzle.res_board):
        # loop over individual cell in the row
        for (box, digit) in zip(cellRow, boardRow):
            # unpack the cell coordinates
            startX, startY, endX, endY, flag = box

            # compute the coordinates of where the digit will be drawn
            # on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.21)
            textX += startX
            textY += endY

            if flag:
                cv2.putText(puzzleImage, str(digit), (textX, textY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    if show_details:
        cv2.imshow('OCR Result', puzzleImage)
        cv2.waitKey(0)

    # solve the sudoku puzzle
    print("[INFO] solving sudoku puzzle...")
    solved = puzzle.solve()
    if solved:
        cnt = float('inf')
        puzzle.show()
    else:
        cnt = puzzle.correct()
        puzzle.show()

    count = 1
    for (cellRow, boardRow) in zip(cellLocs, puzzle.res_board):
        # loop over individual cell in the row
        for (box, digit) in zip(cellRow, boardRow):
            # unpack the cell coordinates
            startX, startY, endX, endY, flag = box

            # compute the coordinates of where the digit will be drawn
            # on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.21)
            textX += startX
            textY += endY

            # draw the corrected puzzle on the sudoku puzzle image
            if flag:
                if count == cnt:
                    cv2.putText(puzzleImage, str(digit), (textX, textY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                else:
                    cv2.putText(puzzleImage, str(digit), (textX, textY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                count += 1
    if show_details and cnt != float('inf'):
            cv2.imshow('corrected', puzzleImage)
            cv2.waitKey(0)

    # loop over the cell locations and board
    count = 1
    for (cellRow, boardRow) in zip(cellLocs, puzzle.res_board):
        # loop over individual cell in the row
        for (box, digit) in zip(cellRow, boardRow):
            # unpack the cell coordinates
            startX, startY, endX, endY, flag = box

            # compute the coordinates of where the digit will be drawn
            # on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.21)
            textX += startX
            textY += endY

            # draw the result digit on the sudoku puzzle image
            if not flag:
                cv2.putText(puzzleImage, str(digit), (textX, textY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if flag:
                if count == cnt:
                    cv2.putText(puzzleImage, str(digit), (textX, textY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                else:
                    cv2.putText(puzzleImage, str(digit), (textX, textY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                count += 1

    # show the output image
    cv2.imshow("Sudoku Result", puzzleImage)
    cv2.waitKey(0)


if __name__ == "__main__":
    solve_sudoku(model_path, image_path, show_details=False)