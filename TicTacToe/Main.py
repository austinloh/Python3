print('Welcome to Tic Tac Toe!')


choice = replay()
while choice == True:
    initial_board = ['', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    instruction_board()
    marker1, marker2 = player_input()
    turn = choose_first()
    print(turn + ' will go first')
    game_on = True
    
    while game_on:
        if turn == 'Player 1':
            display_board(initial_board)
            print('Player 1,')
            position1 = player_choice(initial_board)
            place_marker(initial_board, marker1, position1)
        
            if win_check(initial_board, marker1):
                display_board(initial_board)
                print('Player 1 has won!')
                game_on = False
                choice = replay()
            else:
                if full_board_check(initial_board):
                    display_board(initial_board)
                    print('Tie Game')
                    game_on = False
                    choice = replay()
                else:
                    turn = 'Player 2'
        
        
        else:
            display_board(initial_board)
            print('Player 2,')
            position2 = player_choice(initial_board)
            place_marker(initial_board, marker2, position2)
        
            if win_check(initial_board, marker2):
                display_board(initial_board)
                print('Player 2 has won!')
                game_on = False
                choice = replay()
            else:
                if full_board_check(initial_board):
                    display_board(initial_board)
                    print('Tie Game')
                    game_on = False
                    choice = replay()
                else:
                    turn = 'Player 1'