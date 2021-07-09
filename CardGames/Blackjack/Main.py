#Game Logic

while True:
    print("Welcome to blackjack. To win, the value of your cards should be as close to 21 without going over.\n\
Dealer hits until he reaches 17. Aces count as 1 or 11")
    
    #Creates deck and deal cards
    deck = Deck()
    deck.shuffle()
    
    player_hand = Hand()
    dealer_hand = Hand()
    
    player_hand.add_card(deck.deal())
    dealer_hand.add_card(deck.deal())
    player_hand.add_card(deck.deal())
    dealer_hand.add_card(deck.deal())
    
    #Set up player chips. Default is 100
    player_chips = Chips()
    
    #Take bet
    take_bet(player_chips)
    
    #Show partial cards
    show_some(player_hand, dealer_hand)
    
    playing = True
    while playing:
        
        hit_or_stand(deck, player_hand)
        show_some(player_hand, dealer_hand)
        
        #If player busts, break out of while playing loop
        if player_hand.value > 21:
            player_bust(player_hand, dealer_hand, player_chips)
            break
            
    #Player stands and have not bust. Dealer playing now
    if player_hand.value <= 21:
        while dealer_hand.value < 17:
            hit(deck, dealer_hand)
            
        show_all(player_hand, dealer_hand)

        if dealer_hand.value > 21:
            dealer_bust(player_hand, dealer_hand, player_chips)
            
        elif dealer_hand.value < player_hand.value:
            player_wins(player_hand, dealer_hand, player_chips)
            
        elif dealer_hand.value > player_hand.value:
            dealer_wins(player_hand, dealer_hand, player_chips)
            
        else:
            push(player_hand, dealer_hand)
        
    #Inform player current chips    
    print("\nPlayer current chips are:", player_chips.total)
        
    #Ask to play again
    new_game = input("New game? Enter 'y' or 'n'")
    
    if new_game[0].lower() == 'y':
        continue
        
    else:
        print("Thanks for playing!")
        break
    