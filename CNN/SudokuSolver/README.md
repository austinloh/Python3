## Sudoku Solver
Uses OpenCV to read and process image containing sudoku puzzle. CNN model then used to detect digits. Solves sudoku. Solution overlaid on original image.
Using google colab to run as M1 Macbook as tensorflow not supported \

Credits to https://www.youtube.com/watch?v=qOXDoYUgNlU \
Overall flow: Read image -> grayscale -> gaussian blur -> threshold -> find contours -> select biggest contour -> find transformation matrix -> warp perspective -> split boxes -> predict digits -> solve sudoku -> inverse warp perspective -> overlay

### 1.jpg
Can solve
![](https://github.com/austinloh/Python3/blob/main/CNN/SudokuSolver/Unknown-3.png)

### 2.jpg
Can solve
![](https://github.com/austinloh/Python3/blob/main/CNN/SudokuSolver/Unknown-4.png)

### 3.jpeg
Unable to solve. Wrond detection of digits in puzzle. Might need to process the individual boxes and get the digit contour instead of just using the whole box.
Can check this out. https://becominghuman.ai/part-3-solving-the-sudoku-ai-solver-13f64a090922
![](https://github.com/austinloh/Python3/blob/main/CNN/SudokuSolver/Unknown-2.png)
