from IPython.display import clear_output
    
initial_board = ['', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

def instruction_board():
    print('Instruction Board: ')
    print('1|2|3')
    print('-----')
    print('4|5|6')
    print('-----')
    print('7|8|9')

def display_board(board):
    clear_output()
    print('Current Board: ')
    print(board[1] + '|' + board[2] + '|' + board[3])
    print('-----')
    print(board[4] + '|' + board[5] + '|' + board[6])
    print('-----')
    print(board[7] + '|' + board[8] + '|' + board[9])
    
    
def player_input():
    marker = False
    while marker not in ['X', 'O']:
        marker = input('Player 1, choose X or O: ').upper()
        if marker not in ['X', 'O']:
            print("Invalid choice. Choose from 'X' or 'O'")
    if marker == 'X':
        return ('X', 'O')
    else:
        return ('O', 'X')
    return marker
            
def place_marker(board, marker, position):
    board[position] = marker

def win_check(board, mark):
    return ((board[1] == board[2] == board[3] == mark) or
    (board[4] == board[5] == board[6] == mark) or
    (board[7] == board[8] == board[9] == mark) or
    (board[1] == board[4] == board[7] == mark) or
    (board[2] == board[5] == board[8] == mark) or
    (board[3] == board[6] == board[9] == mark) or
    (board[1] == board[5] == board[9] == mark) or
    (board[3] == board[5] == board[7] == mark))

import random

def choose_first():
    x = random.randint(0,1)
    if x == 0 :
        return 'Player 1'
    else:
        return 'Player 2'
    
def space_check(board, position):
    return board[position] == ' '

def full_board_check(board):
    for i in range(1,10):
        if space_check(board, i):
            return False
    return True

def player_choice(board):
    position = 0
    while position not in [1,2,3,4,5,6,7,8,9] or not space_check(board, position):
        position = int(input('Choose a position: '))
        if position not in [1,2,3,4,5,6,7,8,9] or not space_check(board, position):
            print('Invalid choice. Choose another position')
    return position

def replay():
    choice = input('New Game? Enter Yes or No')
    return choice == 'Yes'