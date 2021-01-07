import time


'''
Class: Sudoku
Input: 9x9 array
function:
    check(): check whether the number can be filled in the sudoku
    solve(): wrapper of solve method
    try(): solving process
    get_param(): get the candidate numbers for a block
    find_unsolved(): find the blocks that need to be filled

variable:
    sudoku: the board of sudoku
    index: how many unsolved blocks have been filled
    unsolved: the list of all unsolved blocks 
'''


class Sudoku():
    def __init__(self, board):
        self.original_board = board.copy()
        self.res_board = board.copy()

    def check(self, row, col, num):
        # check for row
        for item in self.res_board[row]:
            if num == item:
                return False
        # check for column
        for rows in self.res_board:
            if num == rows[col]:
                return False
        # check for 3x3 grid
        x = row // 3 * 3
        y = col // 3 * 3
        for i in range(x, x+3):
            for j in range(y, y+3):
                if num == self.res_board[i][j]:
                    return False
        return True

    def get_candidate(self, x, y):  # get the available numbers in (x, y)
        prem = []
        rows = list(self.res_board[x])
        rows.extend([self.res_board[i][y] for i in range(9)])
        cols = set(rows)
        for i in range(1, 10):
            if i not in cols:
                prem.append(i)
        return prem

    def try_xy(self, x, y):
        p = self.get_candidate(x, y)
        for num in p:
            if self.check(x, y, num):
                self.res_board[x][y] = num
                nx, ny = self.getNext(x, y)
                if nx == -1:  # all blocks are filled
                    return True
                else:
                    res = self.try_xy(nx, ny)
                    if res:
                        return True
                    else:
                        self.res_board[x][y] = 0  # backtrack
        return False

    def getNext(self, x, y):  # find the next unfilled block
        for ny in range(y+1, 9):
            if self.res_board[x][ny] == 0:
                return (x, ny)
        for row in range(x+1, 9):
            for ny in range(0, 9):
                if self.res_board[row][ny] == 0:
                    return (row, ny)
        return (-1, -1)  # no such block exists

    def solve(self):
        # x,y=(0,0) if self.res_board[0][0]==0 else self.getNext(0,0)
        if self.res_board[0][0] == 0:
            flag = self.try_xy(0, 0)
        else:
            x, y = self.getNext(0, 0)
            flag = self.try_xy(x, y)
        return flag

    def get_board(self):
        return self.res_board

    def correct(self):
        count = 1
        for i in range(9):
            for j in range(9):
                if self.original_board[i][j] == 0:
                    continue
                tmp = self.res_board[i][j]
                self.res_board[i][j] = 0
                flag = self.solve()
                if flag:
                    return count
                self.res_board[i][j] = tmp
                count += 1

    def __str__(self):  # print(sovSudoku)
        return '\n'.join(str(i) for i in self.res_board)

    def show(self):
        for i in range(9):
            for j in range(9):
                num = self.res_board[i][j]
                print(num, end=' ')
            print('')
        print('')


# Easy
s1 = [[1, 0, 0, 2, 9, 0, 0, 8, 4],
      [0, 0, 4, 0, 0, 6, 9, 0, 0],
      [2, 5, 0, 7, 0, 0, 0, 0, 3],
      [0, 0, 6, 0, 7, 0, 8, 0, 0],
      [3, 8, 0, 5, 0, 0, 0, 6, 1],
      [0, 0, 2, 0, 8, 3, 5, 0, 0],
      [7, 0, 0, 0, 0, 4, 0, 5, 6],
      [0, 0, 5, 8, 0, 0, 1, 0, 0],
      [6, 4, 0, 0, 5, 7, 0, 0, 8]]

# Medium
s2 = [[6, 0, 0, 0, 0, 0, 7, 0, 0],
      [9, 0, 0, 4, 2, 0, 0, 0, 0],
      [0, 1, 2, 0, 5, 0, 0, 0, 0],
      [5, 0, 0, 8, 0, 9, 0, 0, 0],
      [8, 0, 4, 0, 0, 0, 9, 0, 3],
      [0, 0, 0, 7, 0, 3, 0, 0, 1],
      [0, 0, 0, 0, 7, 0, 2, 8, 0],
      [0, 0, 0, 0, 9, 1, 0, 0, 6],
      [0, 0, 3, 0, 0, 0, 0, 0, 4]]

# Hard
s3 = [[0, 0, 9, 3, 0, 6, 1, 0, 5],
      [0, 0, 5, 7, 0, 4, 0, 0, 3],
      [0, 7, 1, 0, 2, 0, 0, 0, 0],
      [9, 8, 0, 0, 0, 3, 0, 0, 0],
      [0, 0, 0, 6, 0, 8, 0, 0, 0],
      [0, 0, 0, 9, 0, 0, 0, 6, 2],
      [0, 0, 0, 0, 6, 0, 5, 3, 0],
      [2, 0, 0, 4, 0, 7, 9, 0, 0],
      [3, 0, 4, 5, 0, 9, 2, 0, 0]]

'''
s = Sudoku(s3)
begin = time.time()
print(s.solve())
print(s.correct())
s.show()
print(time.time()-begin)
'''