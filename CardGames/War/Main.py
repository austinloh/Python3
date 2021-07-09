player1 = Player('One')
player2 = Player('Two')

new_deck = Deck()
new_deck.shuffle()

for x in range(26):
    player1.add_cards(new_deck.deal_one())
    player2.add_cards(new_deck.deal_one())

game_on = True

round_num = 0

while game_on:
    
    round_num += 1
    print(f'Round {round_num}')
    
    if len(player1.all_cards) == 0:
        print(f'Player {player1.name} out of cards! Player {player2.name} won')
        game_on = False
        break
        
    elif len(player2.all_cards) == 0:
        print(f'Player {player2.name} out of cards! Player {player1.name} won')
        game_on = False
        break
        
    
        
    at_war = True
            
    while at_war:
        player1_cards = []
        player2_cards = []

        player1_cards.append(player1.remove_one())
        player2_cards.append(player2.remove_one())       

        if player1_cards[-1].value > player2_cards[-1].value:
            player1.add_cards(player1_cards)
            player1.add_cards(player2_cards)
            at_war = False

        elif player1_cards[-1].value < player2_cards[-1].value:
            player2.add_cards(player2_cards)
            player2.add_cards(player1_cards)
            at_war = False

        else:
            print('WAR!')

            if len(player1.all_cards) < 5:
                print(f'Player {player1.name} unable to declare war\nPlayer {player2.name} won')
                game_on = False
                break

            elif len(player2.all_cards) < 5:
                print(f'Player {player2.name} unable to declare war\nPlayer {player1.name} won')
                game_on = False
                break
                
            else:
                for num in range(5):
                    player1_cards.append(player1.remove_one())
                    player2_cards.append(player2.remove_one())

